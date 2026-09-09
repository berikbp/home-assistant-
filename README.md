# HVAC RL Control Platform

Reinforcement learning platform for benchmarking HVAC control algorithms against a classical PI controller.

## Quick Start

```bash
# Start the server
cd scripts
python rl_platform.py 8888

# Open in browser
http://localhost:8888
```

---

## How It Works

### The Problem

Buildings use HVAC (Heating, Ventilation, Air Conditioning) systems to maintain comfortable temperatures. The goal is to minimize energy consumption while keeping occupants comfortable.

**The trade-off:**
- More heating/cooling = more comfort, more energy
- Less heating/cooling = less comfort, less energy

### The Solution

Train RL agents to learn optimal control policies that balance comfort and energy.

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Dashboard (HTML/JS)                   │
│  - Training controls (algorithm, hyperparameters)        │
│  - Charts (temperature, power, comfort, actions)         │
│  - Saved agents list                                     │
│  - Occupancy schedule editor                             │
│  - Comfort setpoint sliders                              │
│  - Building parameter sliders                            │
│  - Export buttons                                        │
└──────────────────────┬──────────────────────────────────┘
                       │ HTTP API
┌──────────────────────┴──────────────────────────────────┐
│                  rl_platform.py (Backend)                 │
│  - Thermal model (1R1C building physics)                 │
│  - Gymnasium environment                                 │
│  - SAC / PPO / DDPG training                             │
│  - PI controller (baseline)                              │
│  - Persistent agent storage (models/)                    │
└─────────────────────────────────────────────────────────┘
```

### Data Flow

```
Weather Data (CSV)
      │
      ▼
┌─────────────┐    action    ┌──────────────────┐
│  RL Agent   │ ──────────► │  Thermal Model   │
│  (learned)  │             │  (1R1C physics)   │
│             │ ◄────────── │                   │
└─────────────┘   new_state  └──────────────────┘
      │
      ▼
  Reward = -(discomfort + weight × energy)
```

---

## Workflow

### 1. Configure Settings

In the sidebar:

1. **Select algorithm** — SAC, PPO, or DDPG
2. **Set hyperparameters** — timesteps, learning rate, energy weight, etc.
3. **Set timestep** — how often the agent makes decisions (1-60 minutes)
4. **Set occupancy schedule** — click hours to toggle occupied/unoccupied
5. **Set comfort setpoints** — temperature ranges for occupied/unoccupied
6. **Set building parameters** — thermal model properties

### 2. Train Agent

Click **Train Agent**. The system:

1. Creates a Gymnasium environment with your settings
2. Interpolates weather data to your chosen timestep
3. Runs the RL algorithm for N timesteps
4. Each episode: agent controls HVAC for 14 days (simulated)
5. Agent learns to maximize reward (minimize discomfort + energy)
6. Saves trained model to `models/` with all settings in `meta.json`

### 3. Simulate

Click **Simulate 14 Days**. The system:

1. Runs your trained agent on 14 days of weather
2. Runs the PI controller on the same 14 days
3. Computes metrics for both
4. Displays results on all charts

### 4. Compare

Select saved agents and click **Compare Selected** to overlay multiple agents on the same charts.

---

## The Reward Model (Detailed)

The reward function is the most critical part of the system. It defines **what the agent learns to optimize**. If the reward function is wrong, the agent will learn wrong behavior.

### The Equation

```python
reward = -(discomfort_normalized + energy_weight × energy_normalized)
```

### Step-by-Step Computation

At each timestep, the system computes:

#### Step 1: Check Comfort Band

```
Current state:
  T_indoor = 22.5°C  (current indoor temperature)
  T_low = 21°C       (lower setpoint bound)
  T_high = 24°C      (upper setpoint bound)
```

The comfort band is the temperature range where occupants are comfortable:
- **Occupied hours** (8:00-19:00): 21-24°C
- **Unoccupied hours** (19:00-8:00): 15-30°C (wider range, less strict)

#### Step 2: Compute Discomfort

```python
discomfort = max(T_low - T_indoor, 0) + max(T_indoor - T_high, 0)
```

This measures **how far** the temperature is **outside** the comfort band:

| Scenario | T_indoor | Calculation | Discomfort |
|----------|----------|-------------|------------|
| Too cold | 19°C | max(21-19, 0) + max(19-24, 0) = 2 + 0 | **2.0** |
| Comfortable | 22.5°C | max(21-22.5, 0) + max(22.5-24, 0) = 0 + 0 | **0.0** |
| Too hot | 26°C | max(21-26, 0) + max(26-24, 0) = 0 + 2 | **2.0** |

**Key insight:** Discomfort is always ≥ 0. It's 0 only when temperature is within the band.

#### Step 3: Normalize Discomfort

```python
dn = discomfort / dref
```

Where `dref = 6.8247 / total_steps` is a normalization constant.

**Why normalize?** Without normalization, discomfort (in °C) and energy (in kWh) would be on different scales. The agent would optimize whichever number is larger, ignoring the other.

**What is dref?** It's the maximum possible discomfort over the simulation period. Dividing by it scales discomfort to roughly [0, 1].

#### Step 4: Compute Energy

```python
energy_kwh = (P_heat + P_cool + P_fan) / 1000  # Convert W to kW
energy_normalized = energy_kwh / floor_area / eref
```

Where:
- `P_heat` = heating power (Watts) — positive when heating
- `P_cool` = cooling power (Watts) — positive when cooling
- `P_fan` = fan power (Watts) — always positive when fan is on
- `floor_area` = 48 m² (normalizes to kWh/m²)
- `eref = 4.2263 / total_steps` (normalization constant)

**Why normalize?** Same reason as discomfort — puts energy on [0, 1] scale.

#### Step 5: Combine with Weight

```python
reward = -(dn + energy_weight × en)
```

The `energy_weight` (default 0.6) controls the trade-off:

| energy_weight | Behavior |
|---------------|----------|
| 0.0 | Comfort only — agent ignores energy, always heats/cools to stay in band |
| 0.3 | Comfort-prioritized — mostly comfort, some energy savings |
| 0.6 | Balanced — equal importance (default) |
| 1.0 | Equal weight — comfort and energy equally important |
| 1.5 | Energy-prioritized — agent tolerates some discomfort to save energy |

#### Step 6: Negate

The reward is **negative** because:
- Agent **maximizes** reward
- We want to **minimize** discomfort + energy
- Maximizing negative = minimizing positive

So `reward = -0.5` is better than `reward = -1.0`.

### Complete Example

```
T_indoor = 25°C (too hot by 1°C)
T_low = 21°C, T_high = 24°C
discomfort = max(21-25, 0) + max(25-24, 0) = 0 + 1 = 1.0
dn = 1.0 / dref

P_heat = 0W, P_cool = 2000W, P_fan = 30W
energy = (0 + 2000 + 30) / 1000 / 48 = 0.0423 kWh/m²
en = 0.0423 / eref

reward = -(dn + 0.6 × en)
```

### What the Agent Learns

The agent learns a **policy** — a mapping from state to action:

```
State: [T_indoor, T_outdoor, Solar, Setpoints, Occupancy, Time]
   │
   ▼
Policy (neural network)
   │
   ▼
Action: [fan_speed, supply_temperature]
```

The agent learns:
- **When** to heat/cool (based on time, weather, occupancy)
- **How much** to heat/cool (fan speed)
- **What temperature** to supply (supply temp)
- **When to do nothing** (save energy)

---

## The Thermal Model (Detailed)

The thermal model simulates how a building's temperature changes over time. It's based on **heat transfer physics**.

### The 1R1C Model

```
1R1C = 1 Resistance, 1 Capacitance
```

This is the simplest thermal model that captures essential building dynamics:

```
dT/dt = (Q_wall + Q_solar + Q_hvac) / C
```

Think of it like filling a bathtub:
- **Temperature** = water level
- **C (capacitance)** = bathtub size (bigger = slower to fill/drain)
- **Q_wall** = heat leaking in/out through walls
- **Q_solar** = heat from sunlight
- **Q_hvac** = heat from the HVAC system

### Why Each Parameter Matters

#### Thermal Capacitance (C) — Default: 2.0 K·m²/W

**Physical meaning:** How much energy is needed to change the temperature by 1°C.

**Intuition:**
- **Low C (0.5)** = lightweight building (thin walls, little furniture)
  - Temperature changes quickly
  - Agent must react fast
  - Like a tent — heats up/cools down rapidly

- **High C (10)** = heavy building (thick concrete, lots of thermal mass)
  - Temperature changes slowly
  - Agent can be lazy — temperature inertia helps
  - Like a cave — stays cool in summer, warm in winter

**Effect on control:**
- Low C → agent needs frequent adjustments
- High C → agent can make infrequent, larger adjustments

#### Thermal Resistance (R) — Default: 5.0 K/W

**Physical meaning:** How hard it is for heat to flow through the walls.

**Intuition:**
- **Low R (1)** = poor insulation (old windows, thin walls)
  - Heat flows easily in/out
  - Building loses heat fast in winter
  - Agent must work harder to maintain temperature

- **High R (20)** = excellent insulation (double-glazed windows, thick walls)
  - Heat flows slowly
  - Building retains heat well
  - Agent can save energy — building stays warm longer

**Effect on control:**
- Low R → agent must heat/cool more often
- High R → agent can coast between adjustments

#### Solar Absorptivity (α) — Default: 0.15

**Physical meaning:** Fraction of solar radiation absorbed by the building.

**Intuition:**
- **Low α (0.05)** = reflective surface (white roof, reflective windows)
  - Most sunlight bounces off
  - Less solar heat gain
  - Building stays cooler in summer

- **High α (0.9)** = dark surface (black roof, dark walls)
  - Most sunlight absorbed
  - Significant solar heat gain
  - Building heats up in sun

**Effect on control:**
- Low α → agent relies more on HVAC for heating
- High α → free solar heating during daytime, agent can reduce HVAC

#### Floor Area (A) — Default: 48 m²

**Physical meaning:** The size of the conditioned zone.

**Intuition:**
- **Small A (10 m²)** = single room
  - Less air to heat/cool
  - Faster temperature response
  - Lower total energy

- **Large A (200 m²)** = large open plan
  - More air to heat/cool
  - Slower temperature response
  - Higher total energy

**Effect on control:**
- Small A → faster response, less energy
- Large A → slower response, more energy

#### HVAC Coefficient (K_u) — Default: 0.3 W/(K·m²)

**Physical meaning:** How much heating/cooling power the HVAC system provides per degree of temperature difference.

**Intuition:**
- **Low K_u (0.05)** = weak HVAC (small unit, old system)
  - Can't heat/cool much
  - Takes longer to reach setpoint
  - May struggle in extreme weather

- **High K_u (2.0)** = powerful HVAC (industrial system)
  - Can heat/cool quickly
  - Reaches setpoint fast
  - May overshoot if not careful

**Effect on control:**
- Low K_u → agent must run HVAC longer
- High K_u → agent can use short bursts

### The Heat Transfer Equation

```python
Q_wall = (T_outdoor - T_indoor) / R
```
Heat flowing through walls. Positive when outdoor is warmer (heat flows in).

```python
Q_solar = α × G × A
```
Solar heat gain. G = solar irradiance (W/m²), A = floor area.

```python
Q_hvac = fan × K_u × (T_supply - T_indoor)
```
HVAC heat. Positive when T_supply > T_indoor (heating), negative when cooling.

```python
T_new = T_old + (Q_wall + Q_solar + Q_hvac) / C × dt
```
New temperature after dt hours.

---

## The Evaluation Process (Detailed)

### Overview

Evaluation compares the **trained RL agent** against the **PI baseline** on the same 14-day weather period. Both controllers run independently and their metrics are compared.

### Step-by-Step Process

#### Step 1: Load Weather Data

```
trajectory.csv → 336 hourly rows of:
  - Outdoor temperature (°C)
  - Solar irradiance (W/m²)
```

If timestep < 60 min, weather is interpolated:
```
336 hourly rows × (60 / timestep_min) = total data points
Example: 5-min timestep → 336 × 12 = 4032 data points
```

#### Step 2: Run PI Baseline (No Training)

The PI controller runs deterministically:

```python
for each timestep:
    # 1. Get current state
    T_indoor = current_temperature
    T_setpoint = (T_low + T_high) / 2  # midpoint of comfort band
    
    # 2. Compute error
    error = T_setpoint - T_indoor
    
    # 3. PI control
    fan = Kp × error + Ki × integral_of_error
    fan = clamp(fan, 0, 1)  # bound to [0, 1]
    
    # 4. Set supply temperature
    if error > 0:  # need heating
        T_supply = 35°C
    else:  # need cooling
        T_supply = 16°C
    
    # 5. Step thermal model
    T_new, P_heat, P_cool, P_fan = thermal_model.step(T_indoor, T_outdoor, G, fan, T_supply)
    
    # 6. Record trajectory
    trajectory.append({
        'temperature': T_new,
        'power_heat': P_heat,
        'power_cool': P_cool,
        'power_fan': P_fan,
        ...
    })
```

**Key:** PI uses fixed rules — no learning, no optimization. It's a static benchmark.

#### Step 3: Run RL Agent (Trained)

The trained RL agent runs:

```python
for each timestep:
    # 1. Get current state
    state = [T_indoor, T_outdoor, G, T_low, T_high, occupancy, sin(time), cos(time)]
    
    # 2. Agent picks action (neural network inference)
    action = agent.predict(state)  # returns [fan_raw, tsup_raw]
    
    # 3. Convert action to physical values
    fan = (action[0] + 1) / 2          # [-1,1] → [0,1]
    T_supply = 12 + (action[1] + 1) / 2 * 28  # [-1,1] → [12,40]°C
    
    # 4. Step thermal model (same as PI)
    T_new, P_heat, P_cool, P_fan = thermal_model.step(T_indoor, T_outdoor, G, fan, T_supply)
    
    # 5. Record trajectory
    trajectory.append({...})
```

**Key:** Agent uses learned policy — optimized for reward through training.

#### Step 4: Compute Metrics

After both simulations complete, metrics are computed:

##### Comfort Score

```python
comfort_score = (hours_within_band / total_hours) × 100
```

- Count timesteps where T_indoor is within [T_low, T_high]
- Divide by total timesteps
- Multiply by 100 for percentage

**Example:**
```
4032 timesteps total
3024 timesteps within comfort band
comfort_score = (3024 / 4032) × 100 = 75.0%
```

##### Total Energy

```python
total_energy = sum(P_heat + P_cool + P_fan) × dt / 1000  # kWh
energy_per_m2 = total_energy / floor_area  # kWh/m²
```

##### Total Discomfort

```python
total_discomfort = sum(max(T_low - T, 0) + max(T - T_high, 0))  # K·hours
```

Sum of all temperature violations across the simulation.

##### Total Reward

```python
total_reward = sum(reward_t for each timestep)
```

The cumulative reward over the entire 14-day period. Higher (less negative) is better.

##### Energy Cost

```python
energy_cost = total_energy × electricity_price  # USD
daily_cost = energy_cost / 14  # USD/day
```

##### Fan Speed Average

```python
avg_fan_speed = mean(fan_speed for each timestep)
```

Lower average fan speed generally indicates better control (less HVAC runtime).

#### Step 5: Display Results

Metrics are shown in the dashboard:

| Metric Card | What It Shows |
|-------------|---------------|
| Comfort | % hours within comfort band |
| Energy | Total energy in kWh/m² |
| Total Reward | Cumulative reward (higher = better) |
| Episodes | Number of training episodes completed |
| Energy Cost | Total cost in USD |
| Daily Cost | Average cost per day in USD |

Charts show time-series comparison:

| Chart | RL (solid) | PI (dotted) |
|-------|------------|-------------|
| Temperature | Agent's indoor temp | PI's indoor temp |
| Power | Agent's heating/cooling | PI's heating/cooling |
| Comfort Violations | Agent's violations | PI's violations |
| Actions | Agent's fan + Tsup | PI's fan + Tsup |

### What Makes a "Good" Score?

| Metric | Bad | OK | Good | Excellent |
|--------|-----|-----|------|-----------|
| Comfort | <50% | 60-70% | 70-80% | >80% |
| Energy | >50 kWh/m² | 30-50 | 20-30 | <20 |
| Reward | < -1000 | -500 to -1000 | -200 to -500 | > -200 |

**Note:** Scores depend heavily on building parameters and weather. A "good" score for a poorly insulated building (low R) is different from a well-insulated one (high R).

### Comparing RL vs PI

The RL agent should outperform PI if:
1. It learned the weather patterns (solar cycle, temperature swings)
2. It anticipates occupancy schedule (pre-heats before occupants arrive)
3. It finds energy-saving strategies (coasting, night flushing)
4. It adapts to changing conditions (cloudy vs sunny days)

If PI beats RL, it means:
1. Not enough training timesteps
2. Reward function needs tuning
3. Building parameters are unrealistic
4. Weather data is too simple (agent didn't learn generalization)

---

## Parameters

### Algorithm Selection

| Algorithm | Type | Best For |
|-----------|------|----------|
| **SAC** | Off-policy, maximum entropy | Continuous control, exploration |
| **PPO** | On-policy, clipped | Stable training, simple problems |
| **DDPG** | Off-policy, deterministic | Simple continuous control |

### Training Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Timesteps | 50000 | Total training steps |
| Learning Rate | 0.0003 | Step size for gradient updates |
| Energy Weight | 0.6 | How much to penalize energy (0=comfort only, 1=energy only) |
| Discount Factor (γ) | 0.99 | Value of future rewards (higher = more far-sighted) |
| Tau (τ) | 0.005 | Target network update speed |
| Buffer Size | 50000 | Experience replay memory size |
| Batch Size | 256 | Samples per training update |

### Simulation

| Parameter | Default | Description |
|-----------|---------|-------------|
| Timestep | 5 min | Decision frequency (1-60 min) |

### Occupancy Schedule

24-cell grid. Click to toggle:
- **Cyan** = occupied (comfort setpoints apply)
- **Dark** = unoccupied (wider temperature range allowed)

Default: occupied 8:00-19:00

### Comfort Setpoints

| Parameter | Default | Range |
|-----------|---------|-------|
| Occupied Lower | 21°C | 16-26°C |
| Occupied Upper | 24°C | 20-30°C |
| Unoccupied Lower | 15°C | 10-20°C |
| Unoccupied Upper | 30°C | 25-40°C |

### Building Parameters

| Parameter | Symbol | Default | Unit | Description |
|-----------|--------|---------|------|-------------|
| Thermal Capacitance | C | 2.0 | K·m²/W | How slowly temperature changes |
| Thermal Resistance | R | 5.0 | K/W | How well building insulates |
| Solar Absorptivity | α | 0.15 | - | Fraction of solar radiation absorbed |
| Floor Area | A | 48 | m² | Zone floor area |
| HVAC Coefficient | K_u | 0.3 | W/(K·m²) | Heating/cooling power |

---

## Baseline (PI Controller)

Classical Proportional-Integral controller:

```python
error = T_setpoint - T_indoor
fan = Kp × error  (clamped to [0, 1])
```

Runs deterministically — no training needed. Used as comparison benchmark.

---

## Weather Data

14 days of hourly weather from BOPTEST `bestest_air` test case, interpolated to chosen timestep:

| Timestep | Data Points | Episode Steps |
|----------|-------------|---------------|
| 60 min | 336 | 336 |
| 30 min | 672 | 672 |
| 15 min | 1344 | 1344 |
| 5 min | 4032 | 4032 |
| 1 min | 20160 | 20160 |

---

## Saved Agents

Each trained agent is saved to `models/<name>/` with:
- `model.zip` — serialized SB3 model
- `meta.json` — training config, metrics, timestamps

### Agent Name Format

```
<algo>_<timestamp>
# Example: sac_20260909_143022
```

---

## Charts

| Chart | Shows |
|-------|-------|
| **Temperature** | Indoor (RL vs PI), outdoor, comfort band |
| **Power** | Heating/cooling power (RL vs PI) |
| **Comfort Violations** | Temperature violations per timestep |
| **Actions** | Fan speed and supply temperature (RL vs PI) |
| **Training Progress** | Episode rewards over training |
| **Episode Rewards** | Scatter plot of all episode rewards |

---

## Export

Download data as:
- **RL CSV/JSON** — full trajectory from RL simulation
- **PI CSV** — full trajectory from PI simulation
- **Compare CSV** — side-by-side metrics
- **Training CSV** — episode rewards, comforts, energies

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/train` | POST | Start training |
| `/api/status` | GET | Training progress |
| `/api/results` | GET | Training results |
| `/api/simulate` | POST | Run RL + PI simulation |
| `/api/simulate-saved` | POST | Simulate a saved agent |
| `/api/agents` | GET | List saved agents |
| `/api/agents/compare` | POST | Compare multiple agents |
| `/api/agents/<name>/delete` | DELETE | Delete an agent |
| `/api/sweep` | POST | Hyperparameter sweep |
| `/api/sweep-status` | GET | Sweep progress |
| `/api/export/<kind>` | GET | Export data |
| `/api/building-params` | GET | Get building parameters |

---

## File Structure

```
RL/
├── scripts/
│   ├── rl_platform.py      # Backend (700 lines)
│   ├── dashboard.html       # Frontend (600 lines)
│   └── __init__.py
├── data/
│   └── baseline_14d/
│       └── trajectory.csv   # Weather data (336 hourly rows)
├── models/                  # Trained agents
├── pyproject.toml           # Dependencies
├── .python-version          # Python 3.12
├── .gitignore
├── uv.lock
└── README.md
```

---

## Dependencies

- Python 3.12
- gymnasium
- stable-baselines3
- numpy
- pandas
- plotly.js (CDN in dashboard)
