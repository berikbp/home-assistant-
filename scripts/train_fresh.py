#!/usr/bin/env python3
"""Fresh SAC training with tuned hyperparameters."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from rl_platform import (load_weather, SimpleHVACEnv, SAC, CB, State,
                          save_agent, sim_metrics, run_episode, run_baseline)
from stable_baselines3.common.monitor import Monitor

weather = load_weather(5)
env = Monitor(SimpleHVACEnv(weather, 0.6, timestep_min=5))
st = State()
cb = CB(st)

print("Training SAC 30k steps (tuned LR=5e-5, buffer=100k)...")
m = SAC(
    "MlpPolicy", env,
    learning_rate=5e-5,
    buffer_size=100000,
    batch_size=256,
    gamma=0.99,
    tau=0.005,
    verbose=1,
)
m.learn(total_timesteps=30000, callback=cb)
env.close()
print("Training done.")

rw = [float(x) for x in st.get("rewards")]
cf = [float(x) for x in st.get("comforts")]
en = [float(x) for x in st.get("energies")]
metrics = {
    "episode_rewards": rw, "episode_comforts": cf, "episode_energies": en,
    "avg_reward": sum(rw)/len(rw) if rw else 0,
    "avg_comfort": sum(cf)/len(cf) if cf else 0,
    "avg_energy": sum(en)/len(en) if en else 0,
}
train_cfg = {"algo": "SAC", "timesteps": 30000, "energy_weight": 0.6, "lr": 5e-5,
             "timestep_min": 5, "gamma": 0.99, "tau": 0.005, "buffer_size": 100000, "batch_size": 256}
meta = save_agent("sac_30k_tuned", "SAC", m, metrics, train_cfg)
print(f"Saved: {meta['name']}")

traj = run_episode(m, weather, ew=0.6, timestep_min=5)
rl_m = sim_metrics(traj)
bl_traj = run_baseline(weather, ew=0.6, timestep_min=5)
bl_m = sim_metrics(bl_traj)

print()
print("=" * 60)
print(f"{'Metric':<25} {'Baseline':>12} {'RL Agent':>12} {'Delta':>12}")
print("-" * 60)

b1 = bl_m['comfort_score']
r1 = rl_m['comfort_score']
print(f"{'Comfort Score':<25} {b1:>11.2f}% {r1:>11.2f}% {r1-b1:>+11.2f}%")

b2 = bl_m['total_energy']
r2 = rl_m['total_energy']
print(f"{'Energy (kWh/m2)':<25} {b2:>12.2f} {r2:>12.2f} {r2-b2:>+12.2f}")

b3 = bl_m['total_discomfort']
r3 = rl_m['total_discomfort']
print(f"{'Discomfort (Kh)':<25} {b3:>12.2f} {r3:>12.2f} {r3-b3:>+12.2f}")

b4 = bl_m['total_reward']
r4 = rl_m['total_reward']
print(f"{'Total Reward':<25} {b4:>12.2f} {r4:>12.2f} {r4-b4:>+12.2f}")

bc = bl_m['cost_usd']
rc = rl_m['cost_usd']
print(f"{'Energy Cost':<25} ${bc:>11.2f} ${rc:>11.2f} ${rc-bc:>+11.2f}")

print("=" * 60)
