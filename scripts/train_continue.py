#!/usr/bin/env python3
"""Continue training the best agent to beat baseline on ALL metrics."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from rl_platform import (load_weather, SimpleHVACEnv, SAC, PPO, DDPG, CB, State,
                          save_agent, sim_metrics, run_episode, run_baseline, load_agent_model)
from stable_baselines3.common.monitor import Monitor

# Load existing model
model, meta = load_agent_model("sac_20k_quick")
if model is None:
    print("ERROR: agent not found")
    sys.exit(1)

print(f"Loaded {meta['name']} ({meta['algo']}, {meta['train_cfg']['timesteps']} prev timesteps)")
print(f"Previous metrics: comfort={meta['metrics']['avg_comfort']:.2f}, energy={meta['metrics']['avg_energy']:.2f}")

# Continue training 150k more steps
extra_steps = 30000
weather = load_weather(5)
env = Monitor(SimpleHVACEnv(weather, 0.6, timestep_min=5))
st = State()
cb = CB(st)

model.set_env(env)
print(f"\nContinuing training: {extra_steps} more timesteps...")
model.learn(total_timesteps=extra_steps, callback=cb, reset_num_timesteps=False)
env.close()
print("Training done.")

# Save
rw = [float(x) for x in st.get("rewards")]
cf = [float(x) for x in st.get("comforts")]
en = [float(x) for x in st.get("energies")]
metrics = {
    "episode_rewards": rw, "episode_comforts": cf, "episode_energies": en,
    "avg_reward": sum(rw)/len(rw) if rw else 0,
    "avg_comfort": sum(cf)/len(cf) if cf else 0,
    "avg_energy": sum(en)/len(en) if en else 0,
}
total_ts = meta["train_cfg"]["timesteps"] + extra_steps
train_cfg = {"algo": "SAC", "timesteps": total_ts, "energy_weight": 0.6, "lr": 3e-4,
             "timestep_min": 5, "gamma": 0.99, "tau": 0.005, "buffer_size": 50000, "batch_size": 256}
name = f"sac_{total_ts}_cont"
meta2 = save_agent(name, "SAC", model, metrics, train_cfg)
print(f"Saved: {meta2['name']}")

# Evaluate
traj = run_episode(model, weather, ew=0.6, timestep_min=5)
rl_m = sim_metrics(traj)
bl_traj = run_baseline(weather, ew=0.6, timestep_min=5)
bl_m = sim_metrics(bl_traj)

print("\n" + "="*60)
print(f"{'Metric':<25} {'Baseline':>12} {'RL Agent':>12} {'Delta':>12}")
print("-"*60)
print(f"{'Comfort Score':<25} {bl_m['comfort_score']:>11.2f}% {rl_m['comfort_score']:>11.2f}% {rl_m['comfort_score']-bl_m['comfort_score']:>+11.2f}%")
print(f"{'Energy (kWh/m²)':<25} {bl_m['total_energy']:>12.2f} {rl_m['total_energy']:>12.2f} {rl_m['total_energy']-bl_m['total_energy']:>+12.2f}")
print(f"{'Discomfort (K·h)':<25} {bl_m['total_discomfort']:>12.2f} {rl_m['total_discomfort']:>12.2f} {rl_m['total_discomfort']-bl_m['total_discomfort']:>+12.2f}")
print(f"{'Total Reward':<25} {bl_m['total_reward']:>12.2f} {rl_m['total_reward']:>12.2f} {rl_m['total_reward']-bl_m['total_reward']:>+12.2f}")
blc = f"${bl_m['cost_usd']:.2f}"
rlc = f"${rl_m['cost_usd']:.2f}"
dlc = f"${rl_m['cost_usd']-bl_m['cost_usd']:.2f}"
print(f"{'Energy Cost':<25} {blc:>12} {rlc:>12} {dlc:>12}")
print("="*60)
