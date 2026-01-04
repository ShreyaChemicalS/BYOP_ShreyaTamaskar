import pandas as pd
import numpy as np
import torch
import math
from botorch.models import SingleTaskMultiFidelityGP
from botorch.fit import fit_gpytorch_mll
from botorch.models.cost import AffineFidelityCostModel
from botorch.models.transforms import Standardize, Normalize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.utils.transforms import normalize, unnormalize
from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient
from botorch.acquisition.utils import project_to_target_fidelity
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.optim.optimize import optimize_acqf_mixed
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('newest_dataset_lit.csv')
print(f"Dataset shape: {df.shape}")
print(df.head())

tkwargs = {"dtype": torch.double, "device": "cpu"}

# Extract features - FIXED: Remove 'effective_cat_mol%' since it's calculated
# Also removing duplicates that might cause issues
feature_cols = ['temp_C', 'self_cat', 'external_catalyst_mol%', 'reagent_ratio', 'acid_mol%']
print(f"Using features: {feature_cols}")

X_raw = torch.tensor(df[feature_cols].values, **tkwargs)

# Normalize 
bounds_X = torch.stack([X_raw.min(dim=0).values, X_raw.max(dim=0).values])
X_norm = normalize(X_raw, bounds_X)
print(f"X_norm shape: {X_norm.shape}")

# Get yields
y_LF = torch.tensor(df['yield_%_LF'].values, **tkwargs).unsqueeze(-1)
y_HF = torch.tensor(df['yield_%'].values, **tkwargs).unsqueeze(-1)

print(f"LF yield range: {y_LF.min().item():.2f} to {y_LF.max().item():.2f}")
print(f"HF yield range: {y_HF.min().item():.2f} to {y_HF.max().item():.2f}")

# ============================================================================
# FIXED VIRTUAL LAB
# ============================================================================
class VirtualLab:
    """
    Simulates real experiments with PROPER fidelity relationships
    """
    def __init__(self, X_raw, y_LF, y_HF, noise_level=0.02):
        self.X_raw = X_raw
        self.y_LF = y_LF
        self.y_HF = y_HF
        self.noise_level = noise_level
        self.virtual_experiments = []
        
        # Calculate correlation between LF and HF
        lf_np = y_LF.numpy().flatten()
        hf_np = y_HF.numpy().flatten()
        correlation = np.corrcoef(lf_np, hf_np)[0, 1]
        print(f"LF-HF correlation in data: {correlation:.3f}")
        
        print(f"Virtual Lab initialized with {len(X_raw)} data points")
    
    def simulate_experiment(self, X_full):
        """
        Simulate experiment with PROPER fidelity behavior
        LF (fidelity 0) and HF (fidelity 1) should be CORRELATED
        """
        if X_full.dim() == 1:
            X_full = X_full.unsqueeze(0)
        
        d = X_full.shape[-1] - 1
        X_conditions = X_full[..., :d]
        fidelity = X_full[..., d]  # 0.0 or 1.0
        
        # Denormalize
        X_denorm = unnormalize(X_conditions, bounds_X)
        
        results = []
        for i in range(X_denorm.shape[0]):
            fid = fidelity[i].item()
            x_i = X_denorm[i]
            
            # Find nearest neighbor in original data
            distances = (self.X_raw - x_i).pow(2).sum(dim=-1)
            nearest_idx = distances.argmin().item()
            
            # Get base yields
            lf_base = self.y_LF[nearest_idx].item()
            hf_base = self.y_HF[nearest_idx].item()
            
            # Fidelity-specific simulation
            if fid == 0.0:  # LF
                # LF should be a NOISY version of what HF would be
                # Add systematic bias + noise
                bias = np.random.normal(0, 0.1 * hf_base)  # 10% bias
                noise = np.random.normal(0, self.noise_level * hf_base)
                measured = hf_base + bias + noise
                
            else:  # HF (fidelity 1.0)
                # HF has less noise
                noise = np.random.normal(0, self.noise_level * hf_base)
                measured = hf_base + noise
            
            # Clip and store
            measured = max(0.0, min(100.0, measured))
            
            self.virtual_experiments.append({
                'conditions': x_i.tolist(),
                'fidelity': fid,
                'true_lf': lf_base,
                'true_hf': hf_base,
                'measured': measured
            })
            
            results.append(measured)
        
        return torch.tensor(results, **tkwargs).unsqueeze(-1)
    
    def get_simulation_stats(self):
        if not self.virtual_experiments:
            return "No experiments run yet"
        
        n_lf = sum(1 for exp in self.virtual_experiments if exp['fidelity'] == 0.0)
        n_hf = sum(1 for exp in self.virtual_experiments if exp['fidelity'] == 1.0)
        
        return f"""
        Virtual Lab Stats:
        Total experiments: {len(self.virtual_experiments)}
        LF experiments: {n_lf}
        HF experiments: {n_hf}
        """

# Initialize virtual lab
virtual_lab = VirtualLab(X_raw, y_LF, y_HF, noise_level=0.02)

# ============================================================================
# CREATE TRAINING DATA
# ============================================================================
X_all = []
y_all = []

# Use ALL fidelity levels (0.0 = LF, 1.0 = HF)
for f_level, y in zip([0.0, 1.0], [y_LF, y_HF]):
    f_col = torch.full((X_norm.shape[0], 1), float(f_level), **tkwargs)
    X_f = torch.cat([X_norm, f_col], dim=-1)
    X_all.append(X_f)
    y_all.append(y)

train_x_full = torch.cat(X_all, dim=0)
train_obj = torch.cat(y_all, dim=0)

print(f"Training data shape: {train_x_full.shape}")
print(f"Number of LF points: {(train_x_full[:, -1] == 0.0).sum().item()}")
print(f"Number of HF points: {(train_x_full[:, -1] == 1.0).sum().item()}")

# ============================================================================
# MODEL INITIALIZATION
# ============================================================================
def initialize_model(train_x, train_y):
    d = train_x.shape[-1]
    
    # Define bounds for normalization
    bounds = torch.zeros(2, d, **tkwargs)
    bounds[0] = 0.0
    bounds[1] = 1.0
    bounds[1, -1] = 1.0  # Fidelity dimension max
    
    model = SingleTaskMultiFidelityGP(
        train_x,
        train_y,
        outcome_transform=Standardize(m=1),
        input_transform=Normalize(d=d, bounds=bounds),
        data_fidelities=[d-1]
    )
    
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    return mll, model

# ============================================================================
# COST MODEL
# ============================================================================
def get_cost_model():
    # HF is 10x more expensive than LF
    fidelity_weights = {train_x_full.shape[-1] - 1: 10.0}
    fixed_cost = 1.0
    return AffineFidelityCostModel(fidelity_weights=fidelity_weights, fixed_cost=fixed_cost)

# ============================================================================
# PROJECTION TO HF
# ============================================================================
def make_project_to_HF(fid_index):
    def project(X):
        return project_to_target_fidelity(X=X, target_fidelities={fid_index: 1.0})
    return project

# ============================================================================
# ACQUISITION FUNCTION
# ============================================================================
def get_mfkg(model, train_x, project, cost_model):
    fidelity_index = train_x.shape[-1] - 1
    
    # Get current best from HF data if available
    hf_mask = train_x[:, fidelity_index] == 1.0
    if hf_mask.any():
        hf_points = train_x[hf_mask]
        hf_yields = train_y[hf_mask] if hasattr(train_y, '__iter__') else None
        
        if hf_yields is not None and len(hf_yields) > 0:
            current_best_value = hf_yields.max().item()
        else:
            # Use model prediction
            with torch.no_grad():
                posterior = model.posterior(hf_points)
                current_best_value = posterior.mean.max().item()
    else:
        # Project all to HF and estimate
        hf_points = project(train_x)
        with torch.no_grad():
            posterior = model.posterior(hf_points)
            current_best_value = posterior.mean.max().item()
    
    print(f"Current best value at HF: {current_best_value:.3f}")
    
    cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)
    
    mfkg_acqfn = qMultiFidelityKnowledgeGradient(
        model=model,
        num_fantasies=64,  # Reduced for stability
        current_value=current_best_value,
        cost_aware_utility=cost_aware_utility,
        project=project
    )
    
    return mfkg_acqfn

# ============================================================================
# OPTIMIZATION
# ============================================================================
def optimize_acquisition_function(acq_function, bounds, fidelities):
    dim_input = bounds.shape[1]  # Number of input parameters (without fidelity)
    
    # Create bounds including fidelity dimension
    bounds_full = torch.zeros(2, dim_input + 1, **tkwargs)
    bounds_full[0, :] = 0.0
    bounds_full[1, :] = 1.0
    bounds_full[1, -1] = 1.0  # Fidelity max
    
    # Create fixed features for each fidelity
    fixed_features_list = [{dim_input: float(f)} for f in fidelities]
    
    try:
        best_x_full, acq_value = optimize_acqf_mixed(
            acq_function=acq_function,
            bounds=bounds_full,
            q=1,
            num_restarts=10,  # Reduced for speed
            raw_samples=100,  # Reduced for speed
            options={"batch_limit": 3, "maxiter": 100, "ftol": 1e-4},
            fixed_features_list=fixed_features_list,
        )
        
        # Evaluate at virtual lab
        new_x = best_x_full.detach()
        new_y = virtual_lab.simulate_experiment(new_x)
        
        return new_x, new_y
        
    except Exception as e:
        print(f"Optimization failed: {e}")
        # Fallback: random point
        new_x = torch.rand(1, dim_input + 1, **tkwargs)
        new_x[0, -1] = np.random.choice(fidelities)  # Random fidelity
        new_y = virtual_lab.simulate_experiment(new_x)
        return new_x, new_y

# ============================================================================
# BO LOOP
# ============================================================================
print("\n" + "="*60)
print("STARTING MULTI-FIDELITY BAYESIAN OPTIMIZATION")
print("="*60)

# Initialize with subset of data
n_init = min(20, train_x_full.shape[0])
perm = torch.randperm(train_x_full.shape[0])[:n_init]
train_x = train_x_full[perm].clone()
train_y = train_obj[perm].clone()

print(f"Initial training size: {train_x.shape[0]}")
print(f"Initial HF points: {(train_x[:, -1] == 1.0).sum().item()}")

# Setup
bounds_X_norm = torch.zeros(2, X_norm.shape[1], **tkwargs)
bounds_X_norm[0] = 0.0
bounds_X_norm[1] = 1.0

fidelities_tensor = torch.tensor([0.0, 1.0], **tkwargs)
N_ITER = 10

cost_model = get_cost_model()
project_to_HF = make_project_to_HF(train_x.shape[-1] - 1)

# Track results
results = []
best_hf_yield = y_HF.max().item()  # Best from initial data
best_point = None

for iteration in range(N_ITER):
    print(f"\n--- Iteration {iteration + 1}/{N_ITER} ---")
    
    # 1. Fit model
    try:
        mll, model = initialize_model(train_x, train_y)
        fit_gpytorch_mll(mll)
        
        # 2. Get acquisition function
        mfkg_acqf = get_mfkg(model, train_x, project_to_HF, cost_model)
        
        # 3. Optimize acquisition
        new_x, new_y = optimize_acquisition_function(
            acq_function=mfkg_acqf,
            bounds=bounds_X_norm,
            fidelities=fidelities_tensor,
        )
        
    except Exception as e:
        print(f"Iteration failed: {e}")
        # Use random fallback
        new_x = torch.rand(1, train_x.shape[1], **tkwargs)
        new_x[0, -1] = np.random.choice([0.0, 1.0])
        new_y = virtual_lab.simulate_experiment(new_x)
    
    # 4. Update training data
    train_x = torch.cat([train_x, new_x], dim=0)
    train_y = torch.cat([train_y, new_y], dim=0)
    
    # 5. Record
    chosen_fidelity = new_x[0, -1].item()
    yield_val = new_y.item()
    
    print(f"Query: fidelity {chosen_fidelity:.1f}, yield {yield_val:.3f}%")
    
    # If this was an HF query, check if it's new best
    if chosen_fidelity == 1.0 and yield_val > best_hf_yield:
        best_hf_yield = yield_val
        best_point = new_x.clone()
        print(f"🎯 NEW BEST HF YIELD!")
    
    results.append({
        'iteration': iteration + 1,
        'fidelity': chosen_fidelity,
        'yield': yield_val,
        'best_hf': best_hf_yield
    })

# ============================================================================
# RESULTS
# ============================================================================
print("\n" + "="*60)
print("OPTIMIZATION COMPLETE")
print("="*60)

print(virtual_lab.get_simulation_stats())

# Find best point (prioritize HF points)
hf_mask = train_x[:, -1] == 1.0
if hf_mask.any():
    hf_points = train_x[hf_mask]
    hf_yields = train_y[hf_mask]
    best_idx = hf_yields.argmax()
    best_x = hf_points[best_idx]
    best_y = hf_yields[best_idx].item()
else:
    # No HF points, use LF and project
    best_idx = train_y.argmax()
    best_x = train_x[best_idx]
    best_y = train_y[best_idx].item()
    
    # Project to HF
    best_x_proj = best_x.clone()
    best_x_proj[-1] = 1.0
    # Estimate HF yield using model if available
    try:
        mll, model = initialize_model(train_x, train_y)
        fit_gpytorch_mll(mll)
        with torch.no_grad():
            posterior = model.posterior(best_x_proj.unsqueeze(0))
            best_y = posterior.mean.item()
    except:
        pass

print(f"\nBest HF yield found: {best_y:.3f}%")
print(f"Best from initial data: {y_HF.max().item():.3f}%")
print(f"Improvement: {best_y - y_HF.max().item():.3f}%")

# Denormalize best point
best_params = best_x[:-1].unsqueeze(0)
best_params_denorm = unnormalize(best_params, bounds_X)

print(f"\nOptimal conditions:")
for i, col in enumerate(feature_cols):
    print(f"  {col}: {best_params_denorm[0, i].item():.4f}")

print(f"\nYield progression:")
for r in results:
    print(f"  Iter {r['iteration']:2d}: Fid {r['fidelity']:.1f}, "
          f"Yield {r['yield']:6.3f}%, Best HF: {r['best_hf']:6.3f}%")

# ============================================================================
# VISUALIZATION
# ============================================================================
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Yield progression
ax1 = axes[0]
iterations = [r['iteration'] for r in results]
yields = [r['yield'] for r in results]
fidelities = [r['fidelity'] for r in results]
best_hf = [r['best_hf'] for r in results]

colors = ['red' if f == 0.0 else 'blue' for f in fidelities]
ax1.scatter(iterations, yields, c=colors, s=100, alpha=0.7, edgecolors='black')
ax1.plot(iterations, best_hf, 'g-', linewidth=3, marker='s', markersize=8, label='Best HF')
ax1.axhline(y=y_HF.max().item(), color='r', linestyle='--', label='Initial max')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Yield (%)')
ax1.set_title('MFBO Yield Progression')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Fidelity usage
ax2 = axes[1]
lf_count = sum(1 for f in fidelities if f == 0.0)
hf_count = sum(1 for f in fidelities if f == 1.0)

bars = ax2.bar(['LF (0.0)', 'HF (1.0)'], [lf_count, hf_count], 
               color=['red', 'blue'], alpha=0.7)
ax2.set_xlabel('Fidelity')
ax2.set_ylabel('Number of Queries')
ax2.set_title('Fidelity Usage')

for bar, count in zip(bars, [lf_count, hf_count]):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{count}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("SUCCESSFULLY COMPLETED MFBO!")
print("="*60)