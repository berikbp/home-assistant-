# HVAC RL Presentation Preparation Guide

## 1. PROJECT OVERVIEW

### What You Built
A reinforcement learning platform for HVAC (Heating, Ventilation, and Air Conditioning) control that uses the BOPTEST framework (bestest_air test case) to benchmark RL algorithms against a conventional PI/PID controller.

### Core Problem
Buildings consume ~40% of global energy, with HVAC being the largest contributor. The goal is to find an optimal control policy that minimizes energy consumption while maintaining thermal comfort for occupants.

### Key Trade-off
There is a fundamental tension between:
- **Comfort**: Keeping room temperature within a setpoint band (e.g., 21-24°C during occupied hours)
- **Energy**: Minimizing heating/cooling/fan power consumption

You cannot maximize both simultaneously — improving one degrades the other.

---

## 2. THE REWARD FUNCTION (Critical to Understand)

### Equation
```
reward = -(discomfort_normalized + w_energy * energy_normalized)
```

### Components

**Discomfort (dn):**
```
discomfort_proxy = max(T_lower - T_room, 0) + max(T_room - T_upper, 0)
dn = discomfort_proxy / dref
```
- Measures how far the room temperature deviates from the comfort band
- Only penalizes violations (being too cold OR too hot)
- `dref = 6.8247 / 336` is a normalization factor derived from baseline data
- When temperature is within bounds, discomfort = 0

**Energy (en):**
```
en = (P_heat + P_cool + P_fan) / 1000 / floor_area / eref
```
- Total HVAC power consumption normalized by floor area (48 m²)
- `eref = 4.2263 / 336` is a normalization factor from baseline data
- Converts raw power (Watts) to a comparable scale

**Energy Weight (w_energy):**
- Controls the trade-off between comfort and energy
- w_energy = 0.3: Prioritizes comfort (energy penalty is small)
- w_energy = 0.6: Balanced trade-off
- w_energy = 1.5: Prioritizes energy saving (comfort penalty is relatively smaller)

### Why Negative Reward?
RL algorithms maximize reward. Since we want to MINIMIZE discomfort and energy, we negate the sum. The agent learns to make this value as close to zero as possible (least negative).

### Normalization
Both components are normalized to have comparable magnitudes (~0-10 range for typical episodes). This prevents one component from dominating simply due to scale.

---

## 3. RL ALGORITHMS

### SAC (Soft Actor-Critic)
- **Type**: Off-policy, maximum entropy
- **Key idea**: Maximize expected reward AND entropy (randomness) of the policy
- **Why entropy matters**: Encourages exploration, prevents premature convergence
- **Actions**: Continuous (can output any value in [-1, 1])
- **Best for**: Complex continuous control problems
- **Buffer**: 50,000 transitions (can reuse past experience)

### PPO (Proximal Policy Optimization)
- **Type**: On-policy
- **Key idea**: Update policy carefully using a clipped objective function
- **Why clipping**: Prevents too-large policy updates that destabilize training
- **Actions**: Continuous
- **Best for**: Stable training, less hyperparameter-sensitive
- **Buffer**: Collects fresh experience each update (no replay buffer)

### DDPG (Deep Deterministic Policy Gradient)
- **Type**: Off-policy, deterministic
- **Key idea**: Learn a deterministic policy (always takes the same action for a given state)
- **Actions**: Continuous
- **Best for**: Simple continuous control
- **Caveat**: Less exploration than SAC, can get stuck in local optima

### PID/PI Controller (Baseline)
- Classical control: proportional-integral-derivative
- Uses error signal (setpoint - actual temperature) to compute control action
- Well-understood, tunable, but cannot learn complex patterns
- Your PI controller: Kp=2.0, Ki=0.5

---

## 4. ENVIRONMENT DESIGN

### State Space (8 dimensions)
| Index | Variable | Range | Description |
|-------|----------|-------|-------------|
| 0 | T_indoor | [-20, 60] | Current room temperature (°C) |
| 1 | T_outdoor | [-50, 60] | Outdoor temperature (°C) |
| 2 | G_solar | [0, 1500] | Solar irradiance (W/m²) |
| 3 | T_set_low | [0, 50] | Lower setpoint (°C) |
| 4 | T_set_high | [0, 50] | Upper setpoint (°C) |
| 5 | is_occupied | [0, 20] | Occupied flag (0 or 1) |
| 6 | sin(hour) | [-1, 1] | Time encoding (sin) |
| 7 | cos(hour) | [-1, 1] | Time encoding (cos) |

### Action Space (2 dimensions)
| Index | Action | Range | Mapping |
|-------|--------|-------|---------|
| 0 | Fan speed | [-1, 1] → [0, 1] | fan = (action + 1) / 2 |
| 1 | Supply temp | [-1, 1] → [12, 40] | Tsup = 12 + (action + 1) / 2 * 28 |

### Thermal Model (1R1C)
```
Q_wall = (T_outdoor - T_indoor) / R
Q_solar = alpha * G * A / 1000
Q_hvac = fan * K_u * (T_supply - T_indoor)
T_indoor_new = T_indoor + (Q_wall + Q_solar + Q_hvac) / C * dt
```
- C = 2.0 (thermal capacitance, K·m²/W)
- R = 5.0 (thermal resistance, K/W)
- alpha = 0.15 (solar absorptivity)
- A = 48.0 (floor area, m²)
- K_u = 0.3 (heat transfer coefficient)

### Occupancy Schedule
- Occupied: 8:00-20:00 (setpoints: 21-24°C)
- Unoccupied: 20:00-8:00 (setpoints: 15-30°C)

---

## 5. BOPTEST FRAMEWORK

### What is BOPTEST?
Building Optimization Testing Environment — a simulation platform for testing building control strategies. Uses Modelica/FMU for physics-based simulation.

### Test Case: bestest_air
- Single-zone residential building
- Air-based heating/cooling system
- Includes weather data, occupancy schedules, and energy meters

### Official BOPTEST KPIs
| KPI | Description | Unit |
|-----|-------------|------|
| tdis_tot | Total thermal discomfort | K·h |
| idis_tot | Total indoor air temperature discomfort | K·h |
| ener_tot | Total energy consumption | kWh |
| cost_tot | Total cost | USD |
| emis_tot | Total emissions | kg CO₂ |
| time_rat | Simulation time ratio | - |

### Why BOPTEST?
- Standardized benchmark for fair comparison
- Reproducible results
- Built-in baseline controllers
- Official KPI computation

---

## 6. ML MODELS (Additional Requirement)

### Thermal Comfort Model
- **Goal**: Predict thermal comfort from building/environmental variables
- **Features**: Indoor temp, outdoor temp, solar, humidity, airflow, etc.
- **Target**: Comfort score or predicted mean vote (PMV)
- **Algorithm**: Could be Random Forest, Gradient Boosting, Neural Network
- **Evaluation**: R², MAE, RMSE

### Energy Consumption Model
- **Goal**: Predict energy consumption from building variables
- **Features**: Temperature setpoints, outdoor conditions, occupancy, time
- **Target**: Energy consumption (kWh)
- **Algorithm**: Similar to comfort model
- **Evaluation**: R², MAE, RMSE, MAPE

### Why These Models?
- Can be used as surrogate models for faster evaluation
- Enable model-predictive control (MPC)
- Provide interpretability (feature importance)
- Could replace expensive BOPTEST simulations during training

---

## 7. EXPECTED RESULTS & ANALYSIS

### What to Look For
1. **Learning curves**: Reward should increase (become less negative) over training
2. **Comfort-energy trade-off**: Higher w_energy → lower energy but potentially lower comfort
3. **Algorithm comparison**: SAC typically performs best for continuous control
4. **Stability**: PPO should be most stable, DDPG might show variance

### Red Flags (What NOT to Report as Success)
- Near-zero energy consumption (controller is just doing nothing)
- Constant actions (policy collapsed)
- Very high discomfort with low energy (not a useful trade-off)
- Unstable training (reward oscillating wildly)

### Good Signs
- Smooth learning curve
- Actions that respond to occupancy schedule (heating during occupied hours)
- Temperature maintained within setpoints during occupied hours
- Energy reduced during unoccupied hours

---

## 8. METHODOLOGY DOCUMENTATION

### What to Document
1. Exact BOPTEST configuration (test case, scenario, pricing)
2. Training hyperparameters (learning rate, batch size, etc.)
3. Reward function formula and normalization
4. Episode length and total training steps
5. Evaluation protocol (same environment for all controllers)
6. Random seeds (for reproducibility)

### Reproducibility
- Use the same weather data for all controllers
- Same initial conditions
- Same evaluation period (14 days)
- Same BOPTEST version and configuration

---

## 25 PROBABLE QUESTIONS AND ANSWERS

### Conceptual Questions

**Q1: Why did you choose these three RL algorithms (SAC, PPO, DDPG)?**
A: They represent different paradigms in RL:
- SAC: Off-policy, maximum entropy (best exploration)
- PPO: On-policy, clipped objective (most stable)
- DDPG: Off-policy, deterministic (simplest continuous control)
Together they cover the main approaches to continuous action spaces.

**Q2: What is the advantage of RL over classical PID control?**
A: PID is reactive — it only responds to current error. RL can:
- Learn anticipatory strategies (pre-heat before occupancy)
- Handle nonlinear dynamics better
- Optimize long-term cumulative reward, not just instant error
- Adapt to complex weather patterns and building physics

**Q3: Why use a negative reward instead of positive?**
A: RL maximizes reward. Since we want to minimize discomfort and energy, we negate the sum. The agent learns to make the reward as close to zero as possible (least negative = minimum cost).

**Q4: How does the energy weight (w_energy) affect the learned policy?**
A: It shifts the agent's priority:
- Low w_energy (0.3): Agent focuses on comfort, uses more energy
- High w_energy (1.5): Agent focuses on saving energy, tolerates more discomfort
- The optimal weight depends on the application's priorities

**Q5: Why normalize the reward components?**
A: Without normalization, one component could dominate due to scale. For example, energy in kWh might be ~28 while discomfort is ~6.8. Normalization ensures both contribute equally to the reward signal.

**Q6: What is the role of the thermal model?**
A: It simulates building physics — how indoor temperature responds to:
- Heat loss/gain through walls (Q_wall)
- Solar radiation (Q_solar)
- HVAC heating/cooling (Q_hvac)
The RL agent learns to control HVAC given these dynamics.

**Q7: Why include time encoding (sin/cos of hour) in the state?**
A: Buildings have strong diurnal patterns:
- Solar radiation varies with time of day
- Occupancy follows a schedule
- Outdoor temperature has daily cycles
Time encoding lets the agent learn time-dependent strategies.

**Q8: What is the difference between off-policy and on-policy learning?**
A:
- Off-policy (SAC, DDPG): Can learn from old experience stored in a replay buffer. More sample-efficient.
- On-policy (PPO): Must learn from fresh experience collected by the current policy. Less sample-efficient but more stable.

**Q9: Why does SAC use maximum entropy?**
A: The entropy bonus encourages exploration by making the policy stochastic. This:
- Prevents premature convergence to local optima
- Improves robustness
- Enables better exploration of the state-action space

**Q10: How do you handle the exploration-exploitation trade-off?**
A: 
- SAC: Entropy term naturally handles this
- PPO: Stochastic policy naturally explores
- DDPG: Added noise to actions during training
- All: The reward function implicitly defines what's worth exploring

### Technical Questions

**Q11: Why is the episode length 24 hours?**
A: A 24-hour episode captures a full daily cycle including:
- Occupied and unoccupied periods
- Day and night temperature variations
- Complete solar radiation cycle
This allows the agent to learn daily patterns.

**Q12: How do you ensure fair comparison between controllers?**
A: 
- Same BOPTEST configuration (test case, scenario)
- Same weather data (14-day period)
- Same evaluation metrics
- Same initial conditions
- No dynamic pricing (fixed electricity price)

**Q13: What normalization factors did you use and why?**
A: 
- dref = 6.8247 / 336 (discomfort reference from baseline)
- eref = 4.2263 / 336 (energy reference from baseline)
These are derived from the baseline PI controller's performance, ensuring the RL agent is normalized against a realistic benchmark.

**Q14: Why use a 1R1C thermal model instead of a more complex one?**
A: 
- Captures the essential dynamics (thermal mass + resistance)
- Fast simulation (important for RL training speed)
- Sufficient for demonstrating the concept
- More complex models would slow training without proportional benefit

**Q15: How does the observation space handle variable setpoints?**
A: The setpoints (T_set_low, T_set_high) are included in the observation, allowing the agent to:
- Adapt to different occupancy schedules
- Learn context-dependent strategies
- Handle dynamic setpoint changes

### Results Questions

**Q16: What would you consider a successful result?**
A: The RL controller should:
- Maintain comfort within setpoints during occupied hours
- Reduce energy during unoccupied hours
- Show smooth learning curves
- Demonstrate anticipatory behavior (pre-heating/cooling)
- Outperform PID on the comfort-energy trade-off

**Q17: How do you handle cases where the agent learns to do nothing (near-zero energy)?**
A: This indicates:
- Reward imbalance (energy weight too high)
- The agent found that doing nothing minimizes energy cost
- Need to check reward implementation
- May need to add minimum action constraints

**Q18: What metrics should be reported in the results table?**
A: 
- Comfort Score (% hours within setpoint)
- Total energy consumption (kWh/m²)
- Thermal discomfort (K·h)
- Official BOPTEST KPIs (tdis_tot, ener_tot, etc.)
- Training reward curve
- Evaluation reward

**Q19: How do you determine the "best" configuration?**
A: Not by a single metric. The best configuration:
- Maintains comfort above threshold (e.g., >90%)
- Reduces energy compared to baseline
- Shows stable training
- Is practical for real deployment
The comfort-energy trade-off plot helps identify Pareto-optimal solutions.

**Q20: What is the significance of the occupied vs 24-hour analysis?**
A: It reveals:
- Whether the agent prioritizes occupied hours
- Energy savings during unoccupied periods
- Real-world applicability (buildings are typically occupied ~12h/day)
- Whether the agent learns the occupancy schedule

### Methodology Questions

**Q21: Why not use a more complex reward function?**
A: 
- Simplicity aids interpretation
- Complex rewards can introduce unintended behaviors
- The current formulation clearly separates comfort and energy
- Easier to tune and debug

**Q22: How many training steps are sufficient?**
A: Depends on the algorithm:
- SAC/DDPG: 50,000-100,000 steps typically sufficient
- PPO: May need more due to on-policy nature
- Monitor learning curves — plateaus indicate convergence
- Too few steps → underfitting; too many → overfitting to training weather

**Q23: Why use the same weather data for training and evaluation?**
A: For fair comparison and reproducibility. In practice, you would use different weather periods for training and testing, but for benchmarking, consistency is key.

**Q24: How do the ML models (comfort/energy prediction) complement the RL work?**
A: They provide:
- Interpretability (feature importance)
- Surrogate models for faster evaluation
- Foundation for model-predictive control
- Validation of the simulation model

**Q25: What are the limitations of this study?**
A: 
- Single-zone model (real buildings are multi-zone)
- Simplified thermal model (1R1C)
- Fixed weather period (may not generalize)
- No考虑 of occupancy uncertainty
- No考虑 of equipment degradation
- Computational cost of RL training
- Sim-to-real gap (simulation vs actual building)

---

## PRESENTATION TIPS

### Structure
1. **Problem**: Why HVAC control matters (energy + comfort)
2. **Approach**: RL vs classical control
3. **Methodology**: Reward function, environment, algorithms
4. **Results**: Learning curves, comparison tables, trade-off analysis
5. **Discussion**: What worked, what didn't, limitations
6. **Conclusion**: Key findings and future work

### Key Points to Emphasize
- The trade-off is fundamental — you can't optimize both simultaneously
- Normalization ensures fair comparison
- Multiple algorithms provide robustness to the conclusion
- BOPTEST ensures reproducibility
- The reward weight is the key tuning parameter

### Common Pitfalls to Avoid
- Don't claim RL is "better" without qualification — it depends on the metric
- Don't ignore failed training runs — they're informative
- Don't overstate the simplicity of deployment
- Don't forget to discuss limitations

### Visual Aids
- Learning curves (reward vs episodes)
- Comfort-energy trade-off scatter plot
- Temperature profiles (RL vs PID)
- Action profiles (fan speed, supply temp)
- Comparison tables with all metrics

---

## 9. DEEP RL THEORY (Supervisor Will Probe Here)

### The Bellman Equation
Every RL algorithm is rooted in the Bellman equation:
```
V(s) = max_a [R(s,a) + γ * V(s')]
```
- V(s) = expected cumulative reward from state s
- R(s,a) = immediate reward for taking action a in state s
- γ = discount factor (how much we value future vs immediate reward)
- V(s') = value of the next state

**Why it matters for HVAC**: The agent doesn't just optimize the current step — it considers the long-term consequence of each heating decision. Turning on heating now costs energy but may prevent discomfort later.

### Policy vs Value-Based Methods
| Approach | What it learns | Example |
|----------|---------------|---------|
| Value-based | Value function V(s) or Q(s,a), derive policy from it | DQN (discrete only) |
| Policy-based | Directly learn policy π(a|s) | REINFORCE |
| Actor-Critic | Both — actor = policy, critic = value | SAC, PPO, DDPG |

**Your algorithms are all Actor-Critic**: The "actor" decides what action to take, the "critic" evaluates how good that action was. This hybrid approach combines the stability of value methods with the flexibility of policy methods.

### Why Actor-Critic for HVAC?
- Continuous action space (fan speed, supply temp) → can't use pure value-based (DQN)
- Need stable training → critic provides low-variance gradient estimates
- Need sample efficiency → off-policy variants (SAC, DDPG) reuse experience

### The Policy Gradient Theorem
For policy-based methods, the core update rule:
```
∇J(θ) = E[∇log π_θ(a|s) * A(s,a)]
```
- ∇J(θ) = how to update policy parameters θ
- π_θ(a|s) = probability of action a in state s
- A(s,a) = advantage = how much better than average this action is

**Interpretation**: Increase the probability of actions that have above-average advantage (good actions), decrease probability of below-average actions (bad actions).

### Advantage Estimation (GAE)
PPO uses Generalized Advantage Estimation:
```
A(s,a) = r + γV(s') - V(s)  # 1-step advantage
A(s,a) = r + γr' + γ²r'' + ... - V(s)  # n-step advantage
```
- The "advantage" measures how much better an action is compared to the baseline
- GAE balances bias (1-step) vs variance (n-step) in the estimate
- λ parameter controls the trade-off (typically 0.95-0.99)

### Entropy Regularization (SAC Specific)
SAC's objective:
```
J(π) = E[Σ_t γ^t (r_t + α * H(π(·|s_t)))]
```
- H(π) = entropy of the policy distribution
- α = temperature parameter (auto-tuned in modern SAC)
- Higher entropy → more exploration → more diverse behaviors

**Why this matters**: Without entropy, the policy converges to deterministic (always same action for same state). With entropy, the policy remains stochastic, maintaining exploration even after convergence.

### SAC's Twin Q-Functions
SAC uses two Q-networks (Q1, Q2) and takes the minimum:
```
Q_loss = MSE(Q1(s,a), target) + MSE(Q2(s,a), target)
target = r + γ * min(Q1(s',a'), Q2(s',a'))
```
**Why minimum?** Prevents overestimation bias. If one Q-function overestimates a bad action, the minimum ignores it. This is crucial for stability in continuous control.

### PPO's Clipped Objective
```
L_clip = min(r(θ) * A, clip(r(θ), 1-ε, 1+ε) * A)
```
- r(θ) = probability ratio of new vs old policy
- ε = clip range (typically 0.1-0.2)
- If advantage is positive: don't increase probability too much
- If advantage is negative: don't decrease probability too much

**Why clipping?** Prevents catastrophic policy updates. Without clipping, a single bad gradient step can destroy the policy.

### DDPG's Deterministic Policy
Unlike SAC/PPO which output a distribution (mean + std), DDPG outputs a single action:
```
a = μ_θ(s)  # deterministic policy
```
**Limitation**: Exploration must be added externally (Ornstein-Uhlenbeck noise or Gaussian noise). This is less principled than SAC's entropy-based exploration.

---

## 10. BUILDING PHYSICS DEEP DIVE

### Heat Transfer Fundamentals
Buildings exchange heat through three mechanisms:

**1. Conduction (through walls)**
```
Q_cond = (T_outside - T_inside) / R_total
```
- R_total = sum of all thermal resistances (walls, roof, windows)
- Your model: R = 5.0 K/W (simplified single resistance)

**2. Convection (air movement)**
```
Q_conv = h * A * (T_surface - T_air)
```
- h = convective heat transfer coefficient
- Simplified in your model as part of the HVAC term

**3. Radiation (solar)**
```
Q_solar = α * G * A
```
- α = solar absorptivity (0.15 in your model)
- G = solar irradiance (W/m²)
- A = surface area exposed to sun

### Thermal Capacitance (C)
```
dT/dt = Q_net / C
```
- C = 2.0 K·m²/W in your model
- Higher C → slower temperature response (more thermal mass)
- Lower C → faster response (less thermal mass)
- Real buildings: C depends on wall materials, furniture, air volume

### Your 1R1C Model Explained
```
T_new = T_old + (Q_wall + Q_solar + Q_hvac) / C * dt
```
This is a **lumped parameter model**:
- Assumes uniform temperature throughout the zone
- Single resistance for all heat paths
- Single capacitance for all thermal storage
- Valid when Biot number < 0.1 (lumped vs distributed)

### Why Not a More Complex Model?
| Model | Complexity | Accuracy | Training Speed |
|-------|-----------|----------|----------------|
| 1R1C | Low | Low-Medium | Fast |
| 3R2C | Medium | Medium | Medium |
| 5R3C | High | High | Slow |
| Full FEM | Very High | Very High | Very Slow |

**Trade-off**: For RL training, you need thousands of simulation steps. A complex model would make training prohibitively slow. The 1R1C model captures the essential dynamics for control learning.

### HVAC System Physics
```
Q_hvac = fan * K_u * (T_supply - T_indoor)
```
- fan ∈ [0, 1]: fan speed (0 = off, 1 = max)
- K_u = 0.3: heat transfer coefficient
- T_supply ∈ [12, 40]°C: supply air temperature
- When T_supply > T_indoor: heating
- When T_supply < T_indoor: cooling

**Power consumption**:
```
P_heat = max(Q_hvac, 0) * 1000  # Watts (heating)
P_cool = max(-Q_hvac, 0) * 1000  # Watts (cooling)
P_fan = fan * 50  # Watts (fan motor)
```

### Why This Matters for Results
- If the agent learns low fan + moderate T_sup → low energy but slow response
- If the agent learns high fan + extreme T_sup → fast response but high energy
- The optimal strategy depends on the comfort-energy trade-off weight

---

## 11. BOPTEST INTERNALS

### Architecture
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Controller │ ←→  │   BOPTEST   │ ←→  │   FMU       │
│  (Python)   │ API │   Server    │ API │ (Modelica)  │
└─────────────┘     └─────────────┘     └─────────────┘
```
- **FMU**: Functional Mock-up Unit — packaged Modelica simulation
- **BOPTEST Server**: Manages simulation, provides API
- **Controller**: Your RL agent or PID

### What BOPTEST Provides
1. **Standardized API**: `advance()`, `get_sensor()`, `get_input()`
2. **KPI Computation**: Automatic calculation of official metrics
3. **Scenario Management**: Different weather, occupancy, pricing
4. **Baseline Controllers**: Built-in PID for comparison
5. **Fault Injection**: Test robustness (not used in this study)

### bestest_air Test Case
- **Building**: Single-zone residential
- **HVAC**: Air-based (fan coil unit)
- **Sensors**: Temperature, humidity, power, weather
- **Actuators**: Fan speed, supply temperature
- **Weather**: TMY3 data (typical meteorological year)

### Why BOPTEST Over Custom Simulation?
| Aspect | BOPTEST | Custom Simulation |
|--------|---------|-------------------|
| Validation | Peer-reviewed, benchmarked | Unknown quality |
| KPIs | Standardized, comparable | Ad-hoc metrics |
| Reproducibility | Guaranteed | Depends on implementation |
| Community | Shared knowledge | Isolated |
| Paper acceptance | Recognized framework | May be questioned |

### FMU Co-Simulation
- Modelica compiles to C code → FMU
- FMU runs as a separate process
- Communication via API (JSON over HTTP)
- Enables different time steps for controller vs simulation
- Supports model exchange and co-simulation modes

---

## 12. ML MODEL DESIGN (Comfort & Energy Prediction)

### Thermal Comfort Model

**What is "comfort"?**
- ASHRAE Standard 55: Thermal Environmental Conditions for Human Occupancy
- PMV (Predicted Mean Vote): -3 (cold) to +3 (hot), 0 = neutral
- PPD (Percentage of Dissatisfied People): PMV-based
- Your simplified metric: % hours within setpoint band

**Features for Comfort Model**:
| Feature | Why It Matters |
|---------|---------------|
| T_indoor | Direct measure of thermal state |
| T_outdoor | Drives heat loss/gain |
| G_solar | Solar heat gain |
| Relative Humidity | Affects perceived comfort |
| Air Velocity | Convective cooling |
| Clothing Insulation | Metabolic heat retention |
| Activity Level | Metabolic heat generation |
| Mean Radiant Temperature | Radiant heat exchange |

**Target Variable**:
```
comfort_score = (hours within band) / (total hours)
```
Or continuous: `discomfort = max(T_low - T, 0) + max(T - T_high, 0)`

**Model Options**:
1. **Linear Regression**: Simple, interpretable, baseline
2. **Random Forest**: Handles nonlinearities, feature importance
3. **Gradient Boosting (XGBoost)**: Often best tabular data performance
4. **Neural Network**: Captures complex interactions

**Evaluation Metrics**:
- R² (coefficient of determination): How much variance is explained
- MAE (Mean Absolute Error): Average prediction error
- RMSE (Root Mean Squared Error): Penalizes large errors

### Energy Consumption Model

**Features for Energy Model**:
| Feature | Why It Matters |
|---------|---------------|
| T_set_low, T_set_high | Control targets |
| T_outdoor | Heating/cooling load |
| G_solar | Free cooling/heating |
| Occupancy | Internal heat gains |
| Time of day | Scheduling patterns |
| Previous energy | Autocorrelation |

**Target Variable**:
```
energy_kwh = (P_heat + P_cool + P_fan) * dt / 1000
```

**Model Architecture**:
- Input layer: normalized features
- Hidden layers: 2-3 layers, 64-128 units each
- Output: single value (energy prediction)
- Loss: MSE (Mean Squared Error)
- Optimizer: Adam (lr=0.001)

### Why These Models Matter
1. **Surrogate Modeling**: Replace expensive BOPTEST with fast ML prediction
2. **Interpretability**: Feature importance reveals what drives comfort/energy
3. **Model Predictive Control**: Use predictions to optimize future actions
4. **Transfer Learning**: Pre-train on simulation, fine-tune on real data

---

## 13. EXPERIMENTAL DESIGN DEEP DIVE

### Hyperparameter Tuning Strategy
```
For each algorithm:
  For each w_energy in [0.3, 0.6, 1.5]:
    Train with default hyperparameters
    Evaluate on 14-day period
    Record all metrics
```

**Why these specific weights?**
- 0.3: Comfort-prioritized (energy is "cheap")
- 0.6: Balanced (equal importance)
- 1.5: Energy-prioritized (comfort is "expensive")
- The range spans from comfort-first to energy-first

**Why not more weights?**
- Diminishing returns: The trade-off curve is smooth
- Presentation clarity: 3 points per algorithm is manageable
- Computational budget: Each training takes time

### Training Budget Justification
- 50,000 steps = ~208 episodes × 24 hours/episode
- Sufficient for SAC/PPO to converge on simple problems
- Monitor learning curves: if still improving, extend training
- Too many steps → overfitting to training weather

### Statistical Considerations
**Single Run Limitation**:
- RL has high variance due to random initialization
- Ideally: 5-10 runs with different seeds, report mean ± std
- Practical: At least 2-3 runs to check consistency
- Document: Seed values used, note variance in results

**What to Report**:
- Mean performance across runs
- Standard deviation or confidence intervals
- Best and worst run performance
- Whether differences are practically significant

### Evaluation Protocol
```
1. Train agent with specific w_energy
2. Run trained agent on 14-day evaluation period
3. Record:
   - Hourly temperatures
   - Hourly power consumption
   - Comfort violations
   - BOPTEST KPIs
4. Compare against PID baseline
5. Repeat for each algorithm and weight
```

**Critical**: Same evaluation for all controllers — no cherry-picking.

---

## 14. ADVANCED CONCEPTUAL QUESTIONS (Q26-Q50)

### Deep RL Theory

**Q26: Explain the bias-variance trade-off in value estimation.**
A: 
- **Bias**: Error from approximating the true value function (e.g., using 1-step returns)
- **Variance**: Sensitivity to random fluctuations (e.g., using Monte Carlo returns)
- **Trade-off**: Low bias → high variance (exact but noisy), Low variance → high bias (stable but approximate)
- **GAE (PPO)**: λ parameter controls this trade-off (λ=0 → low variance, λ=1 → low bias)

**Q27: Why do we need two Q-networks in SAC but not in PPO?**
A:
- SAC is off-policy → can overestimate values if using single Q-function
- Twin Q-functions take minimum → prevents overestimation bias
- PPO is on-policy → uses advantage estimation (GAE) instead of Q-values
- GAE has built-in bias reduction through the value function baseline

**Q28: What is the "deadly triad" in RL and how do your algorithms avoid it?**
A: The deadly triad = off-policy + function approximation + bootstrapping → can cause divergence.
- SAC: Uses target networks (slow-updated copy) to stabilize bootstrapping
- DDPG: Also uses target networks
- PPO: On-policy, so avoids the triad entirely
- All: Use experience replay (off-policy) or fresh experience (on-policy) carefully

**Q29: How does the discount factor γ affect the learned behavior?**
A:
- γ = 0: Only care about immediate reward → myopic behavior
- γ = 1: Care equally about all future rewards → far-sighted
- γ = 0.99 (your setting): Balance — care about next ~100 steps
- For HVAC: γ=0.99 means the agent considers ~4 days ahead (100 hours)

**Q30: What is the difference between model-free and model-based RL? Why did you choose model-free?**
A:
- **Model-free**: Learn policy/value directly from experience (SAC, PPO, DDPG)
- **Model-based**: Learn a model of the environment, then plan using it
- **Why model-free**: 
  - Simpler to implement
  - No model学习 error accumulation
  - BOPTEST provides the true model (no need to learn it)
  - More robust to model misspecification

### Building Physics

**Q31: How sensitive are the results to the thermal model parameters (C, R, α)?**
A:
- **C (capacitance)**: Higher C → slower response → agent must act earlier
- **R (resistance)**: Higher R → less heat loss → less energy needed
- **α (absorptivity)**: Higher α → more solar gain → less heating needed during day
- **Sensitivity analysis**: Vary each parameter ±20%, observe impact on optimal policy
- **Practical implication**: Real buildings have different parameters → transfer learning needed

**Q32: What happens if the thermal model is wrong (sim-to-real gap)?**
A:
- **Problem**: RL policy learned on wrong dynamics may fail in reality
- **Mitigation strategies**:
  - Domain randomization (train on varied parameters)
  - System identification (calibrate model to real data)
  - Robust RL (optimize for worst-case dynamics)
  - Sim-to-real transfer (fine-tune on real data)

**Q33: How does solar radiation affect the optimal control strategy?**
A:
- **Daytime + sunny**: Solar heat gain → reduce heating, increase cooling
- **Nighttime**: No solar → rely on insulation, potential pre-cooling
- **Agent should learn**: Anticipate solar patterns, adjust supply temp accordingly
- **If agent ignores solar**: Suboptimal, indicates insufficient training or poor state representation

**Q34: Why is the floor area (48 m²) in the energy normalization?**
A:
- Energy per unit area (kWh/m²) is an intensity metric
- Allows comparison across buildings of different sizes
- Industry standard for energy benchmarking
- Your model: 48 m² represents a single zone or small apartment

### Algorithm Comparison

**Q35: If SAC is "best," why would anyone use PPO or DDPG?**
A:
- **PPO**: More stable, less hyperparameter-sensitive, easier to tune
- **DDPG**: Simpler, faster to train, sufficient for simple problems
- **SAC**: More complex, needs entropy tuning, but better exploration
- **No free lunch**: Problem-dependent — some problems suit PPO better
- **Practical**: PPO is often the default choice for new problems

**Q36: How do the algorithms differ in sample efficiency?**
A:
| Algorithm | Sample Efficiency | Why |
|-----------|------------------|-----|
| SAC | High | Off-policy, replay buffer, reuses experience |
| DDPG | High | Off-policy, replay buffer |
| PPO | Low | On-policy, discards old experience |

**Implication**: PPO needs more environment interaction to reach the same performance.

**Q37: What would happen if you used DQN (discrete RL) instead?**
A:
- DQN outputs discrete actions (e.g., fan: OFF/LOW/MED/HIGH)
- Would need to discretize the action space
- **Problems**:
  - Coarse control → suboptimal performance
  - Curse of dimensionality (many discrete combinations)
  - No smooth policy → jerky control actions
- **Conclusion**: Continuous control (SAC/PPO/DDPG) is more appropriate for HVAC

**Q38: How does the replay buffer size affect SAC/DDPG performance?**
A:
- **Small buffer (10k)**: Fast training, but high correlation between samples
- **Large buffer (100k+)**: More diverse experience, but slower updates
- **Your choice (50k)**: Good balance for HVAC dynamics
- **Too large**: May include very old, irrelevant experience
- **Too small**: May not capture rare but important events

### Reward Design

**Q39: What are the pitfalls of reward shaping?**
A:
- **Reward hacking**: Agent finds unintended ways to maximize reward
- **Example**: If reward penalizes energy, agent learns to do nothing
- **Mitigation**: Careful normalization, constraints, reward clipping
- **Your approach**: Simple, interpretable, hard to hack (comfort + energy)

**Q40: Could you use multi-objective RL instead of scalarized reward?**
A:
- **Multi-objective RL**: Maintain a Pareto front of solutions
- **Scalarized (your approach)**: Combine into single reward with weights
- **Trade-off**:
  - Scalarized: Simple, one policy per weight
  - Multi-objective: More options, but more complex
- **Why scalarized**: Easier to present, compare, and understand

**Q41: What is the "horizon" problem in RL and how does it affect HVAC?**
A:
- **Horizon**: How far into the future the agent considers
- **Short horizon**: Myopic, misses long-term consequences
- **Long horizon**: Far-sighted, but high variance in value estimates
- **HVAC implication**: Heating now affects temperature for hours
- **Your solution**: γ=0.99 gives ~100-step horizon, sufficient for daily patterns

### ML Models

**Q42: Why not combine comfort and energy into a single ML model?**
A:
- **Separate models**: Each focuses on one aspect, easier to interpret
- **Combined model**: May conflate effects, harder to diagnose
- **Practical**: Different stakeholders care about different metrics
- **Methodological**: Clean evaluation of each prediction task

**Q43: How would you handle missing data in the ML models?**
A:
- **Deletion**: Remove incomplete rows (if small fraction)
- **Imputation**: Mean/median/mode for numerical, mode for categorical
- **Advanced**: KNN imputation, MICE (Multiple Imputation by Chained Equations)
- **Prevention**: Ensure BOPTEST provides complete data

**Q44: What is overfitting and how do you prevent it in the ML models?**
A:
- **Overfitting**: Model learns training noise, poor generalization
- **Signs**: High train accuracy, low test accuracy
- **Prevention**:
  - Train/test split (e.g., 80/20)
  - Cross-validation (k-fold)
  - Regularization (L1, L2, dropout)
  - Early stopping
  - Simpler model architecture

**Q45: How do you interpret feature importance in the ML models?**
A:
- **Tree-based models**: Built-in feature importance (Gini, permutation)
- **Linear models**: Coefficient magnitude and sign
- **SHAP values**: Model-agnostic, shows each feature's contribution
- **Partial dependence plots**: Shows marginal effect of each feature
- **Practical**: Reveals what drives comfort/energy → actionable insights

### Experimental Methodology

**Q46: How do you handle the stochasticity in RL training?**
A:
- **Sources**: Random initialization, action sampling, environment dynamics
- **Mitigation**: Fixed random seeds, multiple runs, report variance
- **Practical**: At least 3 runs with different seeds
- **Statistical tests**: t-test, Mann-Whitney U to check significance

**Q47: What is "hyperparameter sensitivity" and how do you assess it?**
A:
- **Definition**: How much performance changes with hyperparameter variation
- **Assessment**: Grid search or random search over key parameters
- **Key hyperparameters**: Learning rate, batch size, network architecture
- **Your approach**: Fixed hyperparameters (reasonable defaults) + sweep over w_energy

**Q48: How do you ensure the results are not just lucky?**
A:
- **Multiple runs**: Same experiment with different seeds
- **Statistical significance**: p-values, confidence intervals
- **Cross-validation**: Evaluate on different weather periods
- **Ablation studies**: Remove components to test their contribution
- **Reproducibility**: Document all settings, share code

**Q49: What is the difference between training performance and evaluation performance?**
A:
- **Training performance**: Reward during learning (includes exploration noise)
- **Evaluation performance**: Deterministic policy on fixed test set
- **Why differ**: Exploration noise removed, no more learning
- **Expected**: Evaluation should be better (no noise) or similar
- **Red flag**: Much worse → overfitting to training conditions

**Q50: How would you extend this work to a real building?**
A:
- **Sim-to-real transfer**: Fine-tune on real data
- **Safety constraints**: Add hard limits on actions
- **Robustness**: Train on varied conditions
- **Gradual deployment**: Start with advisory mode, then autonomous
- **Monitoring**: Real-time performance tracking
- **Fallback**: PID backup if RL fails

---

## 15. COMMON SUPERVISOR CHALLENGES

### "Why should I believe your results?"
- BOPTEST is a recognized benchmark
- Multiple algorithms provide robustness
- Normalization ensures fair comparison
- All code is documented and reproducible

### "What's the novelty here?"
- Systematic comparison of SAC/PPO/DDPG for HVAC
- Reward weight sensitivity analysis
- Integration of ML surrogate models
- Complete benchmarking framework

### "How does this compare to state-of-the-art?"
- Cite recent papers (2023-2024) on RL for HVAC
- Note that SAC is often best for continuous control
- Acknowledge that model-based methods may be more sample-efficient
- Position as foundational work, not necessarily SOTA

### "What are the practical implications?"
- Energy savings potential (X% reduction)
- Comfort improvement potential (Y% increase)
- Scalability to larger buildings
- Integration with existing BMS systems

### "What would you do differently?"
- More hyperparameter tuning
- Multiple random seeds for statistical rigor
- More complex thermal models
- Real-world validation
- Multi-zone extension

---

## 16. KEY FORMULAS REFERENCE CARD

### Reward Function
```
reward = -(dn + w_energy * en)

dn = discomfort_proxy / dref
discomfort_proxy = max(T_low - T_room, 0) + max(T_room - T_high, 0)
dref = 6.8247 / 336

en = (P_heat + P_cool + P_fan) / 1000 / A / eref
eref = 4.2263 / 336
A = 48.0 m²
```

### Thermal Model
```
Q_wall = (T_out - T_in) / R
Q_solar = α * G * A / 1000
Q_hvac = fan * K_u * (T_sup - T_in)
T_in_new = T_in + (Q_wall + Q_solar + Q_hvac) / C * dt

P_heat = max(Q_hvac, 0) * 1000
P_cool = max(-Q_hvac, 0) * 1000
P_fan = fan * 50
```

### State Space
```
obs = [T_in, T_out, G, T_low, T_high, occ, sin(h), cos(h)]
```

### Action Space
```
fan = (a[0] + 1) / 2          # [-1,1] → [0,1]
T_sup = 12 + (a[1] + 1) / 2 * 28  # [-1,1] → [12,40]
```

### BOPTEST KPIs
```
tdis_tot = Σ max(T_low - T, 0) + max(T - T_high, 0)  [K·h]
ener_tot = Σ (P_heat + P_cool + P_fan) * dt / 1000    [kWh]
cost_tot = ener_tot * price                             [USD]
```

---

## 17. LAST-MINUTE CHECKLIST

### Before the Presentation
- [ ] Know the reward equation by heart
- [ ] Know the state and action spaces
- [ ] Know why you chose each algorithm
- [ ] Know the key results (which algorithm won, by how much)
- [ ] Know the limitations
- [ ] Know 2-3 future work directions
- [ ] Practice explaining the trade-off in 30 seconds
- [ ] Prepare back-up slides for deep technical questions

### During the Presentation
- Start with the problem (why HVAC matters)
- Show the approach (RL vs PID)
- Present results clearly (tables + figures)
- Discuss what worked and what didn't
- End with conclusions and future work

### If You Don't Know the Answer
- "That's a great question — I'd need to look into that"
- "My initial hypothesis would be X, but I haven't tested it"
- "That's beyond the scope of this study, but it would be interesting future work"

---

## 18. GLOSSARY OF TERMS

| Term | Definition |
|------|-----------|
| BOPTEST | Building Optimization Testing Environment |
| FMU | Functional Mock-up Unit (packaged Modelica simulation) |
| SAC | Soft Actor-Critic |
| PPO | Proximal Policy Optimization |
| DDPG | Deep Deterministic Policy Gradient |
| PID | Proportional-Integral-Derivative (classical controller) |
| 1R1C | 1-Resistance, 1-Capacitance (thermal model) |
| GAE | Generalized Advantage Estimation |
| Replay Buffer | Storage of past experiences for off-policy learning |
| Entropy | Measure of randomness in policy distribution |
| Advantage | How much better an action is compared to baseline |
| Discount Factor (γ) | Weight for future rewards vs immediate rewards |
| Learning Rate | Step size for parameter updates |
| Batch Size | Number of samples per training update |
| Episode | One complete simulation run (e.g., 24 hours) |
| Timestep | One step within an episode (e.g., 1 hour) |
| Comfort Score | % of hours temperature is within setpoint band |
| Setpoint | Target temperature range (e.g., 21-24°C) |
| Occupied Hours | Hours when building is occupied (e.g., 8am-8pm) |
| Normalization | Scaling rewards to comparable magnitudes |
| Overfitting | Model learns training noise, poor generalization |
| Sim-to-Real Gap | Difference between simulation and real world |
| Pareto Optimal | No objective can be improved without worsening another |
| Sample Efficiency | How much data is needed to learn a good policy |
| Policy Collapse | Agent learns to always take the same action |
| Reward Hacking | Agent finds unintended ways to maximize reward |
