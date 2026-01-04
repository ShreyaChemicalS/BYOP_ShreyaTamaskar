import torch
import pandas as pd
import numpy as np
from botorch.models import SingleTaskMultiFidelityGP
from botorch.models.transforms import Standardize, Normalize
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import UpperConfidenceBound
from botorch.optim.optimize import optimize_acqf_mixed
from botorch.utils.transforms import normalize, unnormalize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from botorch.acquisition import qKnowledgeGradient

tkwargs = {"dtype": torch.double, "device": "cpu"}

# -----------------------------
# 1. Load and preprocess data
# -----------------------------
csv_path = "properly_calibrated_3lf_dataset.csv"
df = pd.read_csv(csv_path)
df = df[:21]

for i in range(len(df)):
    if df['self_cat'].iloc[i] == 0:
       df['effective_cat_mol%'] = df['external_catalyst_mol%']
    else:
       df['effective_cat_mol%'] = df['acid_mol%']

# X: reaction conditions
x_cols = ['temp_K', 'effective_cat_mol%', 'reagent_ratio', 'acid_mol%']  
X_raw = torch.tensor(df[x_cols].values, **tkwargs)

# Y: YIELD values
y_LF1 = torch.tensor(df["yield_lf1"].values, **tkwargs).unsqueeze(-1)
y_LF2 = torch.tensor(df["yield_lf2"].values, **tkwargs).unsqueeze(-1)
y_LF3 = torch.tensor(df["yield_lf3"].values, **tkwargs).unsqueeze(-1)
y_HF  = torch.tensor(df["yield_exp"].values, **tkwargs).unsqueeze(-1)

print("="*60)
print("MFBO WITH BOUNDARY AVOIDANCE")
print("="*60)

# Normalize X
bounds_X = torch.stack([X_raw.min(dim=0).values, X_raw.max(dim=0).values])
X_norm = normalize(X_raw, bounds_X)

# ----------------------------------------------------------
# 2. SIMPLIFIED REALISTIC SIMULATOR
# ----------------------------------------------------------
class RealDataSimulator:
    def __init__(self, X_raw, y_HF, y_LF1, y_LF2, y_LF3):
        self.X_norm = X_norm.numpy()  # Use normalized X
        self.y_HF = y_HF.numpy().flatten()
        self.y_LF1 = y_LF1.numpy().flatten()
        self.y_LF2 = y_LF2.numpy().flatten()
        self.y_LF3 = y_LF3.numpy().flatten()
        
        # Simple interpolation from YOUR data
        from sklearn.neighbors import KNeighborsRegressor
        
        # Train on YOUR actual data
        self.knn_hf = KNeighborsRegressor(n_neighbors=3, weights='distance')
        self.knn_hf.fit(self.X_norm, self.y_HF)
        
        self.knn_lf3 = KNeighborsRegressor(n_neighbors=3, weights='distance')
        self.knn_lf3.fit(self.X_norm, self.y_LF3)
        
        self.knn_lf2 = KNeighborsRegressor(n_neighbors=3, weights='distance')
        self.knn_lf2.fit(self.X_norm, self.y_LF2)
        
        self.knn_lf1 = KNeighborsRegressor(n_neighbors=3, weights='distance')
        self.knn_lf1.fit(self.X_norm, self.y_LF1)
        
        # Store max yields from YOUR data
        self.hf_max = self.y_HF.max()
        self.lf3_max = self.y_LF3.max()
        
        print(f"\nSimulator initialized with YOUR data:")
        print(f"  Max HF yield in data: {self.hf_max:.2f}%")
        print(f"  Max LF3 yield in data: {self.lf3_max:.2f}%")
    
    def simulate_experiment(self, X_full):
        X_conditions = X_full[:, :-1].numpy()  # Already normalized
        fidelity = X_full[:, -1].item()
        
        # Use appropriate KNN for each fidelity
        if fidelity == 1.0:  # HF
            base_yield = self.knn_hf.predict(X_conditions)[0]
            noise = np.random.normal(0, 2.0)  # Small HF noise
            
        elif fidelity == 0.6:  # LF3
            base_yield = self.knn_lf3.predict(X_conditions)[0]
            noise = np.random.normal(0, 5.0)  # Medium LF3 noise
            
        elif fidelity == 0.4:  # LF2
            base_yield = self.knn_lf2.predict(X_conditions)[0]
            noise = np.random.normal(0, 8.0)  # Larger LF2 noise
            
        else:  # LF1 (0.2)
            base_yield = self.knn_lf1.predict(X_conditions)[0]
            noise = np.random.normal(0, 12.0)  # Largest LF1 noise
        
        result = base_yield + noise
        result = max(0.0, min(result, 100.0))  # Clip to 0-100%
        
        return torch.tensor([[result]], **tkwargs)


# Create simulator
simulator = RealDataSimulator(X_raw, y_HF, y_LF1, y_LF2, y_LF3)

# ----------------------------------------------------------
# 3. Global exploration UCB
# ----------------------------------------------------------
class GlobalExplorationUCB(UpperConfidenceBound):
    """UCB with penalty for local points AND bonus for global exploration"""
    
    def __init__(self, model, beta, evaluated_points, current_best_point, iteration, 
                 min_distance=0.1, penalty_weight=5.0, exploration_weight=3.0):
        super().__init__(model, beta)
        self.evaluated_points = evaluated_points
        self.current_best_point = current_best_point
        self.iteration = iteration
        self.min_distance = min_distance
        self.penalty_weight = penalty_weight
        self.exploration_weight = exploration_weight
        
        # Pre-compute some global exploration bonuses
        self.global_hotspots = self._identify_global_hotspots()
    
    def _identify_global_hotspots(self):
        """Identify potential global maxima regions"""
        hotspots = []
        
        # Hotspot 1: Corners (chemical extremes often work)
        hotspots.append(torch.tensor([1.0, 1.0, 1.0, 1.0], **tkwargs))  # All max
        hotspots.append(torch.tensor([0.0, 0.0, 0.0, 0.0], **tkwargs))  # All min
        hotspots.append(torch.tensor([1.0, 0.0, 1.0, 0.0], **tkwargs))  # Alternating
        hotspots.append(torch.tensor([0.0, 1.0, 0.0, 1.0], **tkwargs))  # Alternating
        
        # Hotspot 2: Center region (balanced conditions)
        hotspots.append(torch.tensor([0.5, 0.5, 0.5, 0.5], **tkwargs))
        
        # Hotspot 3: High temp, low catalyst (common pattern)
        hotspots.append(torch.tensor([1.0, 0.2, 0.5, 0.3], **tkwargs))
        
        # Hotspot 4: Low temp, high catalyst
        hotspots.append(torch.tensor([0.2, 1.0, 0.5, 0.7], **tkwargs))
        
        return torch.stack(hotspots)
    
    def forward(self, X):
        # Regular UCB value
        ucb_value = super().forward(X)
        
        # Strategy 1: Penalty for points too close to already evaluated
        if len(self.evaluated_points) > 0:
            penalties = torch.zeros(X.shape[0], device=X.device, dtype=X.dtype)
            
            for i in range(X.shape[0]):
                distances = torch.norm(self.evaluated_points[:, :-1] - X[i:i+1, :-1], dim=1)
                min_dist = distances.min()
                
                if min_dist < self.min_distance:
                    penalty = self.penalty_weight * (1.0 - min_dist/self.min_distance)
                    penalties[i] = penalty
            
            ucb_value = ucb_value - penalties.view(-1, 1)
        
        # Strategy 2: Bonus for exploring GLOBAL hotspots
        exploration_bonuses = torch.zeros(X.shape[0], device=X.device, dtype=X.dtype)
        
        for i in range(X.shape[0]):
            # Calculate distance to each global hotspot
            distances_to_hotspots = torch.norm(self.global_hotspots - X[i:i+1, :-1], dim=1)
            
            # Bonus for being close to any hotspot
            min_hotspot_dist = distances_to_hotspots.min()
            hotspot_bonus = self.exploration_weight * torch.exp(-min_hotspot_dist / 0.3)
            
            # Bonus for being FAR from current best (explore new regions)
            if self.current_best_point is not None:
                dist_to_best = torch.norm(self.current_best_point - X[i:i+1, :-1])
                # More exploration early, less later
                if self.iteration < 15:
                    exploration_bonus = 2.0 * torch.exp(-dist_to_best / 0.5)  # Penalty for being close!
                else:
                    exploration_bonus = 0.5 * torch.exp(-dist_to_best / 0.5)  # Less penalty later
            
            exploration_bonuses[i] = hotspot_bonus
        
        # Add exploration bonuses
        ucb_value = ucb_value + exploration_bonuses.view(-1, 1)
        
        return ucb_value

# ----------------------------------------------------------
# 4. OPTIMIZATION WITH BOUNDARY AVOIDANCE
# ----------------------------------------------------------
def optimize_with_boundary_avoidance(acqf, bounds, fidelity, evaluated_points):
    """Optimize while avoiding boundaries and previously evaluated points"""
    d_input = bounds.shape[1]
    d_full = d_input + 1
    
    bounds_full = torch.zeros(2, d_full, **tkwargs)
    bounds_full[0, :d_input] = 0.0
    bounds_full[1, :d_input] = 1.0
    bounds_full[0, d_full-1] = 0.2
    bounds_full[1, d_full-1] = 1.0
    
    fixed_features_list = [{d_full-1: float(fidelity)}]
    
    # Create initial samples that AVOID boundaries
    raw_samples = 512
    
    # Strategy 1: Random interior points (avoid edges)
    n_interior = raw_samples // 2
    interior_samples = torch.rand(n_interior, d_full, **tkwargs)
    
    # Avoid boundaries: keep points in [0.1, 0.9] range for decision variables
    interior_samples[:, :d_input] = 0.1 + 0.8 * interior_samples[:, :d_input]
    interior_samples[:, d_full-1] = fidelity
    
    # Strategy 2: Points near good existing points
    n_near_good = raw_samples // 4
    if len(evaluated_points) > 0:
        # Find good points (top 20%)
        good_indices = torch.argsort(acqf.evaluated_y.squeeze(), descending=True)[:min(3, len(evaluated_points))]
        good_points = evaluated_points[good_indices]
        
        near_good_samples = torch.randn(n_near_good, d_full, **tkwargs)
        
        for i in range(n_near_good):
            # Pick a random good point as center
            center_idx = torch.randint(0, len(good_points), (1,)).item()
            center = good_points[center_idx]
            
            # Add noise (local search)
            near_good_samples[i] = center + torch.randn_like(center) * 0.15
            near_good_samples[i, d_full-1] = fidelity  # Fix fidelity
            
            # Clip to bounds
            near_good_samples[i, :d_input] = torch.clamp(near_good_samples[i, :d_input], 0.05, 0.95)
    
    else:
        near_good_samples = torch.rand(n_near_good, d_full, **tkwargs)
        near_good_samples[:, :d_input] = 0.1 + 0.8 * near_good_samples[:, :d_input]
        near_good_samples[:, d_full-1] = fidelity
    
    # Strategy 3: Completely random (including some boundaries)
    n_random = raw_samples - n_interior - n_near_good
    random_samples = torch.rand(n_random, d_full, **tkwargs)
    random_samples[:, d_full-1] = fidelity
    
    # Combine all strategies
    raw_x = torch.cat([interior_samples, near_good_samples, random_samples], dim=0)
    
    try:
        best_x, acq_value = optimize_acqf_mixed(
            acq_function=acqf,
            bounds=bounds_full,
            q=1,
            num_restarts=15,
            raw_samples=raw_x,
            options={
                "maxiter": 200,
                "ftol": 1e-4,
                "disp": False
            },
            fixed_features_list=fixed_features_list,
        )
        
        return best_x, acq_value
        
    except Exception as e:
        print(f"Optimization failed: {e}")
        # Fallback: pick a random interior point
        random_point = torch.rand(1, d_full, **tkwargs)
        random_point[:, :d_input] = 0.3 + 0.4 * random_point[:, :d_input]
        random_point[:, -1] = fidelity
        return random_point, torch.tensor([[0.0]], **tkwargs)

# ----------------------------------------------------------
# 4.5 NEW OPTIMIZATION FUNCTION WITH EXPLOITATION
# ----------------------------------------------------------
def optimize_with_exploitation(acqf, bounds, fidelity, evaluated_points, current_best_point, iteration):
    d_input = bounds.shape[1]
    d_full = d_input + 1
    
    bounds_full = torch.zeros(2, d_full, **tkwargs)
    bounds_full[0, :d_input] = 0.0
    bounds_full[1, :d_input] = 1.0
    bounds_full[0, d_full-1] = 0.2
    bounds_full[1, d_full-1] = 1.0
    
    fixed_features_list = [{d_full-1: float(fidelity)}]
    
    # Create initial samples - focus on exploitation
    raw_samples = 512
    
    # Dynamic exploitation ratio
    if iteration < 10:
        exploitation_ratio = 0.5
    elif iteration < 20:
        exploitation_ratio = 0.7
    else:
        exploitation_ratio = 0.8
    
    # Strategy 1: Points NEAR CURRENT BEST
    n_near_best = int(raw_samples * exploitation_ratio)
    near_best_samples = torch.randn(n_near_best, d_full, **tkwargs)
    
    # Create current best point with fidelity dimension
    current_best_full = torch.cat([current_best_point, 
                                   torch.tensor([[fidelity]], **tkwargs)], dim=-1)
    
    for i in range(n_near_best):
        perturbation = torch.randn_like(current_best_full) * 0.15
        near_best_samples[i] = current_best_full + perturbation
        near_best_samples[i, d_full-1] = fidelity
        near_best_samples[i, :d_input] = torch.clamp(near_best_samples[i, :d_input], 0.05, 0.95)
    
    # Strategy 2: Keep your existing near_good strategy
    n_near_good = raw_samples // 4
    if len(evaluated_points) > 0:
        good_indices = torch.argsort(acqf.evaluated_y.squeeze(), descending=True)[:min(3, len(evaluated_points))]
        good_points = evaluated_points[good_indices]
        
        near_good_samples = torch.randn(n_near_good, d_full, **tkwargs)
        
        for i in range(n_near_good):
            center_idx = torch.randint(0, len(good_points), (1,)).item()
            center = good_points[center_idx]
            near_good_samples[i] = center + torch.randn_like(center) * 0.15
            near_good_samples[i, d_full-1] = fidelity
            near_good_samples[i, :d_input] = torch.clamp(near_good_samples[i, :d_input], 0.05, 0.95)
    else:
        near_good_samples = torch.rand(n_near_good, d_full, **tkwargs)
        near_good_samples[:, :d_input] = 0.1 + 0.8 * near_good_samples[:, :d_input]
        near_good_samples[:, d_full-1] = fidelity
    
    # Strategy 3: Random - FIXED THE NEGATIVE DIMENSION ERROR
    n_random = max(0, raw_samples - n_near_best - n_near_good)  # Ensure non-negative
    if n_random > 0:
        random_samples = torch.rand(n_random, d_full, **tkwargs)
        random_samples[:, d_full-1] = fidelity
    else:
        random_samples = torch.zeros(0, d_full, **tkwargs)
    
    # Combine
    raw_x = torch.cat([near_best_samples, near_good_samples, random_samples], dim=0)
    
    try:
        best_x, acq_value = optimize_acqf_mixed(
            acq_function=acqf,
            bounds=bounds_full,
            q=1,
            num_restarts=15,
            raw_samples=raw_x,
            options={"maxiter": 200, "ftol": 1e-4, "disp": False},
            fixed_features_list=fixed_features_list,
        )
        return best_x, acq_value
    except Exception as e:
        print(f"  Optimization note: {str(e)[:50]}...")
        random_point = torch.rand(1, d_full, **tkwargs)
        random_point[:, :d_input] = 0.3 + 0.4 * random_point[:, :d_input]
        random_point[:, -1] = fidelity
        return random_point, torch.tensor([[0.0]], **tkwargs)

# ----------------------------------------------------------
# 5. BO LOOP WITH DIVERSITY
# ----------------------------------------------------------
# Initial training data
# Initial training data - USE YOUR REAL DATA!
print("\nUsing YOUR ACTUAL DATA for initial training...")
fidelity_levels = [0.2, 0.4, 0.6, 1.0]
X_all = []
Y_all = []

# Use ALL your 21 data points at all 4 fidelities
for i in range(len(X_norm)):
    # LF1 (fidelity 0.2)
    X_lf1 = torch.cat([X_norm[i:i+1], torch.tensor([[0.2]], **tkwargs)], dim=-1)
    X_all.append(X_lf1)
    Y_all.append(y_LF1[i:i+1])
    
    # LF2 (fidelity 0.4)
    X_lf2 = torch.cat([X_norm[i:i+1], torch.tensor([[0.4]], **tkwargs)], dim=-1)
    X_all.append(X_lf2)
    Y_all.append(y_LF2[i:i+1])
    
    # LF3 (fidelity 0.6)
    X_lf3 = torch.cat([X_norm[i:i+1], torch.tensor([[0.6]], **tkwargs)], dim=-1)
    X_all.append(X_lf3)
    Y_all.append(y_LF3[i:i+1])
    
    # HF (fidelity 1.0)
    X_hf = torch.cat([X_norm[i:i+1], torch.tensor([[1.0]], **tkwargs)], dim=-1)
    X_all.append(X_hf)
    Y_all.append(y_HF[i:i+1])

train_x_full = torch.cat(X_all, dim=0)
train_obj = torch.cat(Y_all, dim=0)

print(f"Initial training data: {len(train_x_full)} points (YOUR REAL 21×4 = 84 data points)")
print(f"Best HF yield in your data: {y_HF.max().item():.2f}%")
print(f"Best LF3 yield in your data: {y_LF3.max().item():.2f}%")

# Main BO loop
N_ITER = 30
results = []
best_overall = y_HF.max().item()

# ADDED: Track current best point for exploitation
current_best_point = X_norm[y_HF.argmax()].unsqueeze(0)  # Start with best from data
current_best_yield = best_overall

# ADDED: Track promising points for HF conversion
promising_points_queue = []
hf_queries_used = 0
max_hf_queries = 8
lf_threshold = 70.0  # LF yield threshold for HF conversion

bounds_X_norm = torch.zeros(2, X_norm.shape[1], **tkwargs)
bounds_X_norm[0] = 0.0
bounds_X_norm[1] = 1.0

print("\n" + "="*60)
print("STARTING DIVERSITY-PROMOTING MFBO WITH HF CONVERSION")
print("="*60)
print(f"LF threshold for HF conversion: {lf_threshold}%")
print(f"Max HF queries: {max_hf_queries}")

for it in range(N_ITER):
    print(f"\n--- Iteration {it+1}/{N_ITER} ---")
    print(f"HF queries used: {hf_queries_used}/{max_hf_queries}")
    print(f"Promising points in queue: {len(promising_points_queue)}")
    
    # CHECK: If we have promising points and HF budget, convert one to HF
    if promising_points_queue and hf_queries_used < max_hf_queries:
        lf_point, lf_yield, lf_fidelity = promising_points_queue.pop(0)
        
        print(f"\nCONVERTING promising LF point to HF...")
        print(f"LF yield: {lf_yield:.2f}% (fidelity {lf_fidelity})")
        
        # Create HF version
        hf_point = lf_point.clone()
        hf_point[:, -1] = 1.0
        
        # Evaluate at HF
        hf_yield = simulator.simulate_experiment(hf_point)
        hf_queries_used += 1
        
        print(f"HF yield: {hf_yield.item():.2f}%")
        
        # Update training data
        train_x_full = torch.cat([train_x_full, hf_point], dim=0)
        train_obj = torch.cat([train_obj, hf_yield], dim=0)
        
        # Track results
        results.append({
            'iteration': it+1,
            'fidelity': 1.0,
            'yield': hf_yield.item(),
            'point': hf_point[0].tolist(),
            'converted_from_lf': True,
            'original_lf_yield': lf_yield,
            'original_lf_fidelity': lf_fidelity
        })
        
        # Update best
        if hf_yield.item() > best_overall:
            best_overall = hf_yield.item()
            print(f"NEW BEST HF: {best_overall:.3f}")
        
        # Update current best point for exploitation
        if hf_yield.item() > current_best_yield:
            current_best_point = hf_point[:, :-1].clone()
            current_best_yield = hf_yield.item()
            print(f"Updated focus point to this HF location")
        
        # Skip regular iteration since we did HF conversion
        continue
    
    # Regular iteration (LF exploration)
    # Dynamic fidelity selection
    if it < 10:
        fidelity = 0.2  # Start with cheap LF
    elif it < 20:
        fidelity = 0.4 if it % 3 == 0 else 0.2  # Mix of fidelities
    else:
        fidelity = 0.6  # Higher fidelity later
    
    # Fit model
    d = train_x_full.shape[-1]
    bounds = torch.zeros(2, d, dtype=train_x_full.dtype, device=train_x_full.device)
    bounds[0] = 0.0
    bounds[1, :-1] = 1.0
    bounds[1, -1] = 1.0
    
    model = SingleTaskMultiFidelityGP(
        train_x_full,
        train_obj,
        outcome_transform=Standardize(m=1),
        data_fidelities=[d-1],
    )
    
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    # Debug: Check GP is working
    print(f"  GP trained on {len(train_x_full)} points")
    print(f"  Training yields range: [{train_obj.min().item():.2f}, {train_obj.max().item():.2f}]")

    # Current best
    hf_mask = train_x_full[:, -1] == 1.0
    if hf_mask.any():
        current_best = train_obj[hf_mask].max().item()
        # ADDED: Update current best point
        current_best_idx = train_obj[hf_mask].argmax()
        current_best_point = train_x_full[hf_mask][current_best_idx:current_best_idx+1, :-1]
        current_best_yield = current_best
    else:
        current_best = train_obj.max().item()
        # ADDED: Update current best point
        current_best_idx = train_obj.argmax()
        current_best_point = train_x_full[current_best_idx:current_best_idx+1, :-1]
        current_best_yield = current_best
    
    print(f"Current best: {current_best:.3f}")
    print(f"Current focus point yield: {current_best_yield:.3f}")
    print(f"Chosen fidelity: {fidelity:.2f}")
    
    # Create acquisition function with diversity penalty
    # Dynamic beta: explore more early, exploit more late
    dynamic_beta = 3.0 if it < 15 else 1.5
    
    # Create acquisition function with diversity penalty AND global exploration
    dynamic_beta = 3.0 if it < 15 else 1.5

    acqf = qKnowledgeGradient(
       model=model,
       num_fantasies=64,  # Number of fantasy points for lookahead
       current_value=current_best,  # Current best observed value
    )
    
    # Store evaluated y for the acquisition function
    acqf.evaluated_y = train_obj.clone()
    
    # REPLACED: Optimize with exploitation focus instead of boundary avoidance
    # Optimize with exploitation focus
    try:
        new_x, acq_value = optimize_with_exploitation(
            acqf=acqf,
            bounds=bounds_X_norm,
            fidelity=fidelity,
            evaluated_points=train_x_full,
            current_best_point=current_best_point,
            iteration=it
        )
    except Exception as e:
        print(f"KG optimization failed, using simple UCB fallback: {e}")
        # Fallback to simple UCB
        from botorch.acquisition import UpperConfidenceBound
        acqf_fallback = UpperConfidenceBound(model, beta=2.0)
        new_x, acq_value = optimize_with_exploitation(
            acqf=acqf_fallback,
            bounds=bounds_X_norm,
            fidelity=fidelity,
            evaluated_points=train_x_full,
            current_best_point=current_best_point,
            iteration=it
        )
    
    
    is_boundary = torch.any(new_x[:, :-1] < 0.01) or torch.any(new_x[:, :-1] > 0.99)
    if is_boundary:
        print(f"WARNING: Point at boundary: {new_x}")
        # Try to move it inward
        new_x[:, :-1] = torch.clamp(new_x[:, :-1], 0.05, 0.95)
    
   
    true_yield = simulator.simulate_experiment(new_x)
    yield_val = true_yield.item()
    
    print(f"Optimized point: {new_x}")
    print(f"TRUE YIELD: {yield_val:.3f}")
    
    
    train_x_full = torch.cat([train_x_full, new_x], dim=0)
    train_obj = torch.cat([train_obj, true_yield], dim=0)
    
    print(f"Query: fidelity {fidelity:.2f}, yield {yield_val:.3f}")
    
   
    if fidelity < 1.0 and yield_val >= lf_threshold and hf_queries_used < max_hf_queries:
        print(f"✓ LF yield > {lf_threshold}% - Adding to HF conversion queue")
        promising_points_queue.append((new_x.clone(), yield_val, fidelity))
    
    # Track results
    results.append({
        'iteration': it+1,
        'fidelity': fidelity,
        'yield': yield_val,
        'point': new_x[0].tolist(),
        'converted_from_lf': False
    })
    
    # Update best
    if yield_val > best_overall and fidelity >= 0.6:  # Only trust higher fidelities
        best_overall = yield_val
        print(f"NEW BEST: {best_overall:.3f}")
    
    
    if yield_val > current_best_yield and fidelity >= 0.6:
        current_best_point = new_x[:, :-1].clone()
        current_best_yield = yield_val
        print(f"  Updated focus point to this location")

# ----------------------------------------------------------
# 6. FINAL BATCH HF VALIDATION (convert remaining promising points)
# ----------------------------------------------------------
print("\n" + "="*60)
print("FINAL BATCH VALIDATION PHASE")
print("="*60)

if promising_points_queue and hf_queries_used < max_hf_queries:
    print(f"\nValidating remaining {len(promising_points_queue)} promising points at HF...")
    
    for i, (lf_point, lf_yield, lf_fidelity) in enumerate(promising_points_queue):
        if hf_queries_used >= max_hf_queries:
            break
        
        # Create HF version
        hf_point = lf_point.clone()
        hf_point[:, -1] = 1.0
        
        # Evaluate at HF
        hf_yield = simulator.simulate_experiment(hf_point)
        hf_queries_used += 1
        
        print(f"  Point {i+1}: LF={lf_yield:.2f}% → HF={hf_yield.item():.2f}%")
        
        # Update training data
        train_x_full = torch.cat([train_x_full, hf_point], dim=0)
        train_obj = torch.cat([train_obj, hf_yield], dim=0)
        
        # Track results
        results.append({
            'iteration': N_ITER + i + 1,
            'fidelity': 1.0,
            'yield': hf_yield.item(),
            'point': hf_point[0].tolist(),
            'converted_from_lf': True,
            'original_lf_yield': lf_yield,
            'original_lf_fidelity': lf_fidelity
        })
        
        # Update best
        if hf_yield.item() > best_overall:
            best_overall = hf_yield.item()

# ----------------------------------------------------------
# 7. RESULTS
# ----------------------------------------------------------
print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)

# Find best HF yield
hf_results = [r for r in results if r['fidelity'] == 1.0]
converted_results = [r for r in hf_results if r.get('converted_from_lf', False)]

if hf_results:
    best_hf = max(hf_results, key=lambda x: x['yield'])
    print(f"\nBest HF yield found: {best_hf['yield']:.3f}")
    
    if best_hf.get('converted_from_lf', False):
        print(f"  Converted from LF{best_hf.get('original_lf_fidelity', '?')}: {best_hf.get('original_lf_yield', 'N/A')}%")
    
    # Denormalize best point
    best_point_norm = torch.tensor(best_hf['point'][:-1], **tkwargs).unsqueeze(0)
    best_conditions = unnormalize(best_point_norm, bounds_X)
    
    print(f"\nBest conditions:")
    for i, col in enumerate(x_cols):
        print(f"  {col}: {best_conditions[0, i].item():.4f}")
else:
    # Use best overall
    best_overall_result = max(results, key=lambda x: x['yield'])
    print(f"\nBest overall yield: {best_overall_result['yield']:.3f} (fidelity {best_overall_result['fidelity']:.2f})")

print(f"\nMaximum in original data: {y_HF.max().item():.3f}")
print(f"Best found by BO: {best_overall:.3f}")
print(f"Improvement: {best_overall - y_HF.max().item():.3f}")

# Conversion statistics
print(f"\nConversion Statistics:")
print(f"  LF evaluations: {sum(1 for r in results if r['fidelity'] < 1.0)}")
print(f"  HF evaluations: {len(hf_results)}")
print(f"  Points converted to HF: {len(converted_results)}")
print(f"  HF queries used: {hf_queries_used}/{max_hf_queries}")

if converted_results:
    print(f"\nLF-to-HF conversion success:")
    successful = sum(1 for r in converted_results if r['yield'] > y_HF.max().item())
    print(f"  Successful improvements: {successful}/{len(converted_results)}")

print(f"\nYield progression (converted points only):")
for r in converted_results[-10:]:
    print(f"  Iter {r['iteration']}: LF{r.get('original_lf_fidelity', '?')}={r.get('original_lf_yield', 'N/A')}% → HF={r['yield']:.3f}%")

print(f"\nBoundary analysis:")
boundary_points = 0
interior_points = 0
for r in results:
    point = torch.tensor(r['point'][:-1], **tkwargs)
    if torch.any(point < 0.01) or torch.any(point > 0.99):
        boundary_points += 1
    else:
        interior_points += 1

print(f"  Boundary points: {boundary_points}")
print(f"  Interior points: {interior_points}")
print(f"  Boundary percentage: {boundary_points/len(results)*100:.1f}%")

