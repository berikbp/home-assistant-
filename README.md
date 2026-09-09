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

## Why This Project Exists

**Context:** Buildings consume ~40% of global energy. HVAC systems account for ~50% of building energy use. Traditional controllers (PID) are simple but inefficient — they react to temperature changes without anticipating future conditions.

**The opportunity:** RL agents can learn to:
- Anticipate weather changes (pre-heat before cold front arrives)
- Exploit solar gains (reduce HVAC when sun is heating the building)
- Optimize for occupancy patterns (pre-cool before people arrive)
- Balance comfort vs energy in ways humans can't manually tune

**This platform** lets you train, evaluate, and compare RL agents against a classical baseline in a simulated building environment.

---

## How It Works

### The Problem

Buildings use HVAC (Heating, Ventilation, Air Conditioning) systems to maintain comfortable temperatures. The goal is to minimize energy consumption while keeping occupants comfortable.

**The trade-off:**
- More heating/cooling = more comfort, more energy
- Less heating/cooling = less comfort, less energy

**Why this is hard:** The optimal control depends on:
- Current weather (outdoor temp, solar radiation)
- Building properties (insulation, thermal mass)
- Occupancy schedule (when people are present)
- Time of day (solar cycle, temperature swings)
- Future conditions (will it get hotter or colder?)

A PI controller only reacts to current error. An RL agent can learn to anticipate.

### The Solution

Train RL agents to learn optimal control policies that balance comfort and energy.

**How RL works (simplified):**
1. Agent observes state (indoor temp, outdoor temp, solar, time, occupancy)
2. Agent takes action (fan speed, supply temperature)
3. Environment responds (new temperature, energy consumed)
4. Agent receives reward (negative of discomfort + energy)
5. Agent updates its policy to get higher reward next time
6. Repeat for thousands of episodes until policy converges

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

**Why this architecture?**
- Single-file backend (rl_platform.py) — easy to understand, modify, deploy
- Dashboard in HTML/JS — no frontend build step, runs in any browser
- HTTP API — frontend and backend are decoupled, can be extended
- Gymnasium interface — standard RL API, works with any algorithm

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

**Why this flow?**
- Agent and thermal model are decoupled — can swap either one
- Weather drives the simulation — same weather for fair comparison
- Reward signal is the only learning mechanism — agent never sees the physics equations

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

**Why configure before training?** Different buildings need different parameters. A concrete office (high C, high R) behaves differently from a glass house (low C, low R). The agent must be trained on the specific building it will control.

### 2. Train Agent

Click **Train Agent**. The system:

1. Creates a Gymnasium environment with your settings
2. Interpolates weather data to your chosen timestep
3. Runs the RL algorithm for N timesteps
4. Each episode: agent controls HVAC for 14 days (simulated)
5. Agent learns to maximize reward (minimize discomfort + energy)
6. Saves trained model to `models/` with all settings in `meta.json`

**What happens during training?**
- The agent explores: tries random actions to discover what works
- The agent exploits: uses known good actions more often
- Over time, the agent's policy improves — average reward increases
- Training is complete when reward stabilizes or timesteps are exhausted

**Why 14 days?** Long enough to capture weather variation (sunny/cloudy/cold/warm days), short enough to train quickly.

### 3. Simulate

Click **Simulate 14 Days**. The system:

1. Runs your trained agent on 14 days of weather
2. Runs the PI controller on the same 14 days
3. Computes metrics for both
4. Displays results on all charts

**Why simulate after training?** Training uses exploration noise (agent tries random actions). Simulation uses the learned policy without noise — shows true performance.

**Why run PI too?** To have a baseline for comparison. If RL can't beat PI, the agent hasn't learned anything useful.

### 4. Compare

Select saved agents and click **Compare Selected** to overlay multiple agents on the same charts.

**Why compare?** To see which algorithm/hyperparameters work best for your building.

---

## The Reward Model (Detailed)

The reward function is the most critical part of the system. It defines **what the agent learns to optimize**. If the reward function is wrong, the agent will learn wrong behavior.

**Context:** In RL, the agent has no inherent understanding of "comfort" or "energy." It only sees numbers. The reward function translates human goals (keep people comfortable, save energy) into a single number the agent can maximize.

### The Equation

```python
reward = -(discomfort_normalized + energy_weight × energy_normalized)
```

### Step-by-Step Computation

At each timestep, the system computes:

#### Step 1: Determine the Comfort Band

```python
def _sch(self, s):
    h = s % 24
    if h in self.occ_hours:      # occupied hours (8:00-19:00)
        return (21, 24, 1)       # band = [21°C, 24°C], occupied=1
    return (15, 30, 0)           # band = [15°C, 30°C], occupied=0
```

The comfort band changes based on time of day:
- **Occupied hours** (8:00-19:00): 21-24°C — strict, narrow band
- **Unoccupied hours** (19:00-8:00): 15-30°C — relaxed, wide band

**Why different bands?**
- When people are present, comfort matters most — narrow band
- When building is empty, save energy — allow wider temperature range
- This mimics real building operation (setback at night)

#### Step 2: Compute Discomfort

```python
dis = max(lo - self.Tin, 0) + max(self.Tin - hi, 0)
```

This measures **how far** the temperature is **outside** the comfort band:

| Scenario | T_indoor | Calculation | Discomfort |
|----------|----------|-------------|------------|
| Too cold | 19°C | max(21-19, 0) + max(19-24, 0) = 2 + 0 | **2.0** |
| Comfortable | 22.5°C | max(21-22.5, 0) + max(22.5-24, 0) = 0 + 0 | **0.0** |
| Too hot | 26°C | max(21-26, 0) + max(26-24, 0) = 0 + 2 | **2.0** |

**Key insight:** Discomfort is always ≥ 0. It's 0 only when temperature is within the band.

**Why this formula?**
- `max(lo - T, 0)` captures cold violations only (ignores if T > lo)
- `max(T - hi, 0)` captures hot violations only (ignores if T < hi)
- Summing them gives total violation regardless of direction
- Linear penalty: 2°C violation is twice as bad as 1°C violation

#### Step 3: Normalize Discomfort

```python
dn = dis / dref
```

Where `dref = 6.8247 / total_steps` is a normalization constant.

**Why normalize?** Without normalization, discomfort (in °C) and energy (in kWh) would be on different scales. The agent would optimize whichever number is larger, ignoring the other.

**What is dref?** It's derived from the maximum possible discomfort over the simulation period. Dividing by it scales discomfort to roughly [0, 1].

**Example:**
- Without normalization: discomfort = 2.0, energy = 0.04
- Agent would focus entirely on discomfort (bigger number)
- With normalization: dn = 0.3, en = 0.2
- Agent treats both similarly

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

**Why kWh/m²?** Energy per unit area is an intensity metric. Allows comparison across buildings of different sizes.

#### Step 5: Combine with Weight

```python
reward = -(dn + energy_weight × en)
```

The `energy_weight` (default 0.6) controls the trade-off:

| energy_weight | Behavior | When to use |
|---------------|----------|-------------|
| 0.0 | Comfort only — agent ignores energy | Comfort-critical (hospitals) |
| 0.3 | Comfort-prioritized — mostly comfort | Offices during work hours |
| 0.6 | Balanced — equal importance | General purpose (default) |
| 1.0 | Equal weight — comfort and energy | Energy-conscious buildings |
| 1.5 | Energy-prioritized — tolerates discomfort | Low-priority spaces (storage) |

**Why this weighting?**
- Without weight, comfort and energy have different magnitudes
- Weight lets you tune the trade-off for your specific needs
- Different buildings need different weights (hospital vs warehouse)

#### Step 6: Negate

The reward is **negative** because:
- Agent **maximizes** reward
- We want to **minimize** discomfort + energy
- Maximizing negative = minimizing positive

So `reward = -0.5` is better than `reward = -1.0`.

**Why not minimize directly?** Most RL algorithms are designed for maximization. Negating is the standard convention.

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

**Example learned behaviors:**
- Morning: pre-heat building before occupants arrive
- Sunny afternoon: reduce HVAC (solar gains provide free heating)
- Night: let temperature drift (wider unoccupied band)
- Cold front: increase heating before temperature drops

---

## The Thermal Model (Detailed)

The thermal model simulates how a building's temperature changes over time. It's based on **heat transfer physics**.

**Context:** Real buildings are complex — hundreds of thermal zones, multiple HVAC systems, varying occupancy. This model is a simplification that captures the essential dynamics while being fast enough for RL training.

### Why a Simplified Model?

| Model | Accuracy | Training Speed | Use Case |
|-------|----------|----------------|----------|
| 1R1C (this) | Low-Medium | Fast | RL training, prototyping |
| 3R2C | Medium | Medium | Detailed simulation |
| 5R3C | High | Slow | Research, validation |
| Full FEM | Very High | Very Slow | Engineering design |

**Trade-off:** For RL, you need thousands of simulation steps. A complex model would make training prohibitively slow. The 1R1C model captures the essential dynamics for control learning.

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

**Real-world examples:**
- C ≈ 0.5: Temporary structure, tent, thin-walled building
- C ≈ 2: Typical residential house
- C ≈ 5: Concrete office building
- C ≈ 10+: Heavy industrial building, underground structure

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

**Real-world examples:**
- R ≈ 1: Old building, single-pane windows, no insulation
- R ≈ 3: Typical residential house
- R ≈ 5: Modern house with basic insulation
- R ≈ 10+: Passive house, heavily insulated building

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

**Real-world examples:**
- α ≈ 0.1: White reflective roof
- α ≈ 0.15: Light-colored building (default)
- α ≈ 0.5: Gray concrete
- α ≈ 0.9: Black roof, dark facades

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

**Why this matters:**
- Energy is normalized by area (kWh/m²)
- But absolute energy scales with area
- A 48 m² zone uses less total energy than a 200 m² zone

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

**Real-world examples:**
- K_u ≈ 0.05: Small window AC unit
- K_u ≈ 0.3: Typical residential HVAC
- K_u ≈ 1.0: Commercial rooftop unit
- K_u ≈ 2.0+: Industrial system

### The Heat Transfer Equation

```python
Q_wall = (T_outdoor - T_indoor) / R
```
Heat flowing through walls. Positive when outdoor is warmer (heat flows in).

**Why this formula?** Fourier's law of heat conduction: heat flow = ΔT / R.

```python
Q_solar = α × G × A
```
Solar heat gain. G = solar irradiance (W/m²), A = floor area.

**Why this formula?** Solar gain = absorptivity × irradiance × exposed area.

```python
Q_hvac = fan × K_u × (T_supply - T_indoor)
```
HVAC heat. Positive when T_supply > T_indoor (heating), negative when cooling.

**Why this formula?** Convective heat transfer: Q = m × cp × ΔT. Simplified as K_u × ΔT.

```python
T_new = T_old + (Q_wall + Q_solar + Q_hvac) / C × dt
```
New temperature after dt hours.

**Why this formula?** Energy balance: ΔT = Q_total × dt / C. Basic thermodynamics.

---

## The Evaluation Process (Detailed)

### Overview

Evaluation compares the **trained RL agent** against the **PI baseline** on the same 14-day weather period. Both controllers run independently and their metrics are compared.

**Context:** In research, you need to prove your method works. Comparing against a baseline (PI) shows whether RL actually improves performance.

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

**Why interpolate?** Real weather data is hourly. But HVAC can update more frequently (5 min, 15 min). Interpolation provides realistic weather at finer resolution.

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

**Why PI as baseline?**
- Simple, well-understood, widely used
- No training required — deterministic performance
- Represents "standard" building control
- Easy to beat (but not always!)

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

**Why the same thermal model?** Fair comparison. Both controllers see the same physics.

#### Step 4: Compute Metrics

After both simulations complete, metrics are computed:

##### Comfort Score

```python
# Count timesteps inside comfort band
comf = ((room >= lo) & (room <= hi)).astype(float)

# Compute percentage
comfort_score = 100 × comf.mean()
```

**Example:**
```
4032 timesteps total
3024 timesteps within comfort band
comfort_score = 100 × 3024/4032 = 75.0%
```

**Why percentage?** Easy to understand, comparable across different simulation lengths.

##### Occupied Comfort Score

```python
# Only count timesteps when building is occupied
occupied_mask = [t["is_occupied"] for t in traj]
occ_comf = comf[occupied_mask == 1]

# Compute percentage for occupied hours only
occupied_comfort_score = 100 × occ_comf.mean()
```

**Why separate metric?** Occupied comfort matters more than unoccupied comfort. A building that's 75% comfortable overall but 50% comfortable during occupied hours is failing.

##### Total Energy

```python
total_energy = sum(P_heat + P_cool + P_fan) × dt / 1000  # kWh
energy_per_m2 = total_energy / floor_area  # kWh/m²
```

**Why kWh/m²?** Normalizes by building size. Allows comparison across different buildings.

##### Total Discomfort

```python
total_discomfort = sum(max(T_low - T, 0) + max(T - T_high, 0))  # K·hours
```

Sum of all temperature violations across the simulation.

**Why this metric?** Shows total "human suffering" — how much time was spent outside comfort zone.

##### Total Reward

```python
total_reward = sum(reward_t for each timestep)
```

The cumulative reward over the entire 14-day period. Higher (less negative) is better.

**Why track reward?** It's what the agent optimized for. Shows how well the agent learned.

##### Energy Cost

```python
energy_cost = total_energy × electricity_price  # USD
daily_cost = energy_cost / 14  # USD/day
```

**Why track cost?** Translates energy to dollars. Makes results actionable for building owners.

##### Fan Speed Average

```python
avg_fan_speed = mean(fan_speed for each timestep)
```

Lower average fan speed generally indicates better control (less HVAC runtime).

**Why track fan speed?** Shows HVAC utilization. Lower = less wear and tear on equipment.

#### Step 5: Display Results

Metrics are shown in the dashboard:

| Metric Card | What It Shows | Why It Matters |
|-------------|---------------|----------------|
| Comfort | % hours within comfort band | Primary goal — keep people comfortable |
| Energy | Total energy in kWh/m² | Cost driver — lower is better |
| Total Reward | Cumulative reward | Agent's objective — higher is better |
| Episodes | Training progress | Shows how much training was done |
| Energy Cost | Total cost in USD | Business metric — money saved/lost |
| Daily Cost | Average cost per day | Budgeting metric — daily operating cost |

Charts show time-series comparison:

| Chart | RL (solid) | PI (dotted) | Why It Matters |
|-------|------------|-------------|----------------|
| Temperature | Agent's indoor temp | PI's indoor temp | Shows comfort performance |
| Power | Agent's heating/cooling | PI's heating/cooling | Shows energy usage |
| Comfort Violations | Agent's violations | PI's violations | Shows where comfort failed |
| Actions | Agent's fan + Tsup | PI's fan + Tsup | Shows control strategy |

### What Makes a "Good" Score?

| Metric | Bad | OK | Good | Excellent |
|--------|-----|-----|------|-----------|
| Comfort | <50% | 60-70% | 70-80% | >80% |
| Energy | >50 kWh/m² | 30-50 | 20-30 | <20 |
| Reward | < -1000 | -500 to -1000 | -200 to -500 | > -200 |

**Note:** Scores depend heavily on building parameters and weather. A "good" score for a poorly insulated building (low R) is different from a well-insulated one (high R).

**Context:** These benchmarks are based on typical HVAC performance. Real buildings vary widely.

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

**Why RL should win (in theory):**
- PI is reactive — only responds to current error
- RL is proactive — anticipates future conditions
- PI has fixed gains — same response regardless of context
- RL has adaptive policy — different response for different situations

**Why PI sometimes wins:**
- PI is simple — less room for error
- RL is complex — can overfit to training data
- PI is well-tuned — decades of engineering knowledge
- RL is untrained — not enough timesteps to learn

---

## Parameters

### Algorithm Selection

| Algorithm | Type | Best For | Why |
|-----------|------|----------|-----|
| **SAC** | Off-policy, maximum entropy | Continuous control, exploration | Automatically tunes exploration, good for complex tasks |
| **PPO** | On-policy, clipped | Stable training, simple problems | Simple, reliable, hard to break |
| **DDPG** | Off-policy, deterministic | Simple continuous control | Fast, but less exploratory |

**Context:**
- **Off-policy** = can reuse old experience (more sample efficient)
- **On-policy** = must use fresh experience (less sample efficient, but simpler)
- **Maximum entropy** = encourages exploration (avoids getting stuck)
- **Clipped** = prevents large policy updates (stable training)
- **Deterministic** = always picks the same action for same state (less exploration)

### Training Hyperparameters

| Parameter | Default | Description | Effect if Wrong |
|-----------|---------|-------------|-----------------|
| Timesteps | 50000 | Total training steps | Too few = underfit, too many = overfit |
| Learning Rate | 0.0003 | Step size for gradient updates | Too high = unstable, too low = slow |
| Energy Weight | 0.6 | Comfort vs energy trade-off | Too high = uncomfortable, too low = wasteful |
| Discount Factor (γ) | 0.99 | Future vs immediate reward | Too low = myopic, too high = far-sighted |
| Tau (τ) | 0.005 | Target network update speed | Too high = unstable, too low = slow learning |
| Buffer Size | 50000 | Experience replay memory | Too small = limited experience, too large = old data |
| Batch Size | 256 | Samples per training update | Too small = noisy, too large = slow |

**Context:** Hyperparameters control how the agent learns. Wrong settings can prevent learning entirely.

### Simulation

| Parameter | Default | Description | Why It Matters |
|-----------|---------|-------------|----------------|
| Timestep | 5 min | Decision frequency (1-60 min) | Finer = more realistic, coarser = faster training |

**Context:** Real HVAC systems update every 5-15 minutes. Shorter timesteps are more realistic but require more computation.

### Occupancy Schedule

24-cell grid. Click to toggle:
- **Cyan** = occupied (comfort setpoints apply)
- **Dark** = unoccupied (wider temperature range allowed)

Default: occupied 8:00-19:00

**Context:** Occupancy is the primary driver of comfort requirements. Buildings use setback schedules to save energy when empty.

### Comfort Setpoints

| Parameter | Default | Range | Why |
|-----------|---------|-------|-----|
| Occupied Lower | 21°C | 16-26°C | ASHRAE Standard 55 comfort zone |
| Occupied Upper | 24°C | 20-30°C | ASHRAE Standard 55 comfort zone |
| Unoccupied Lower | 15°C | 10-20°C | Allow temperature drift when empty |
| Unoccupied Upper | 30°C | 25-40°C | Allow temperature drift when empty |

**Context:** ASHRAE Standard 55 defines thermal comfort conditions. Typical office comfort is 21-24°C.

### Building Parameters

| Parameter | Symbol | Default | Unit | Description | Physical Meaning |
|-----------|--------|---------|------|-------------|------------------|
| Thermal Capacitance | C | 2.0 | K·m²/W | How slowly temperature changes | Building thermal mass |
| Thermal Resistance | R | 5.0 | K/W | How well building insulates | Wall insulation quality |
| Solar Absorptivity | α | 0.15 | - | Fraction of solar radiation absorbed | Surface color/material |
| Floor Area | A | 48 | m² | Zone floor area | Room size |
| HVAC Coefficient | K_u | 0.3 | W/(K·m²) | Heating/cooling power | HVAC system capacity |

**Context:** These parameters define the building. Different buildings need different parameters. The agent must be trained on the specific building it will control.

---

## Baseline (PI Controller)

Classical Proportional-Integral controller:

```python
error = T_setpoint - T_indoor
fan = Kp × error + Ki × integral(error)
fan = clamp(fan, 0, 1)
```

**Why PI as baseline?**
- Simple, well-understood, widely used in HVAC
- No training required — deterministic performance
- Represents "standard" building control
- Easy to beat (but not always!)

**PI limitations:**
- Reactive — only responds to current error
- Fixed gains — same response regardless of context
- No anticipation — doesn't predict future conditions
- No optimization — doesn't minimize energy

---

## Weather Data

14 days of hourly weather from BOPTEST `bestest_air` test case, interpolated to chosen timestep:

| Timestep | Data Points | Episode Steps | Training Time |
|----------|-------------|---------------|---------------|
| 60 min | 336 | 336 | Fast |
| 30 min | 672 | 672 | Medium |
| 15 min | 1344 | 1344 | Medium |
| 5 min | 4032 | 4032 | Slow |
| 1 min | 20160 | 20160 | Very Slow |

**Context:** Weather drives the simulation. Same weather ensures fair comparison between controllers.

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

**Why save meta.json?** To know what settings were used for training. Critical for reproducibility.

---

## Charts

| Chart | Shows | Why It Matters |
|-------|-------|----------------|
| **Temperature** | Indoor (RL vs PI), outdoor, comfort band | Primary comfort metric |
| **Power** | Heating/cooling power (RL vs PI) | Primary energy metric |
| **Comfort Violations** | Temperature violations per timestep | Shows when comfort failed |
| **Actions** | Fan speed and supply temperature (RL vs PI) | Shows control strategy |
| **Training Progress** | Episode rewards over training | Shows learning progress |
| **Episode Rewards** | Scatter plot of all episode rewards | Shows variance and convergence |

---

## Export

Download data as:
- **RL CSV/JSON** — full trajectory from RL simulation
- **PI CSV** — full trajectory from PI simulation
- **Compare CSV** — side-by-side metrics
- **Training CSV** — episode rewards, comforts, energies

**Why export?** For further analysis, paper writing, or presentation.

---

## API Endpoints

| Endpoint | Method | Description | When to Use |
|----------|--------|-------------|-------------|
| `/api/train` | POST | Start training | When training an agent |
| `/api/status` | GET | Training progress | During training |
| `/api/results` | GET | Training results | After training |
| `/api/simulate` | POST | Run RL + PI simulation | To evaluate trained agent |
| `/api/simulate-saved` | POST | Simulate a saved agent | To re-evaluate old agent |
| `/api/agents` | GET | List saved agents | To see available agents |
| `/api/agents/compare` | POST | Compare multiple agents | To compare algorithms |
| `/api/agents/<name>/delete` | DELETE | Delete an agent | To clean up |
| `/api/sweep` | POST | Hyperparameter sweep | To find best settings |
| `/api/sweep-status` | GET | Sweep progress | During sweep |
| `/api/export/<kind>` | GET | Export data | To download results |
| `/api/building-params` | GET | Get building parameters | To check current settings |

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
- gymnasium — standard RL interface
- stable-baselines3 — RL algorithm implementations
- numpy — numerical computing
- pandas — data manipulation
- plotly.js (CDN in dashboard) — interactive charts
