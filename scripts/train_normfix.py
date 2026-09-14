#!/usr/bin/env python3
"""Fresh SAC with fixed reward normalization."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from rl_platform import (load_weather, SimpleHVACEnv, SAC, CB, State,
                          save_agent, sim_metrics, run_episode, run_baseline)
from stable_baselines3.common.monitor import Monitor

weather = load_weather(5)
env = Monitor(SimpleHVACEnv(weather, 0.6, timestep_min=5))
st = State()
cb = CB(st)

print("Training SAC 30k steps (fixed normalization)...")
m = SAC("MlpPolicy", env, learning_rate=3e-4, buffer_size=100000,
        batch_size=256, gamma=0.99, tau=0.005, verbose=1)
m.learn(total_timesteps=30000, callback=cb)
env.close()
print("Training done.")

rw = [float(x) for x in st.get("rewards")]
cf = [float(x) for x in st.get("comforts")]
en = [float(x) for x in st.get("energies")]
metrics = {"episode_rewards": rw, "episode_comforts": cf, "episode_energies": en,
           "avg_reward": sum(rw)/len(rw) if rw else 0,
           "avg_comfort": sum(cf)/len(cf) if cf else 0,
           "avg_energy": sum(en)/len(en) if en else 0}
save_agent("sac_30k_normfix", "SAC", m, metrics,
           {"algo":"SAC","timesteps":30000,"energy_weight":0.6,"lr":3e-4,
            "timestep_min":5,"gamma":0.99,"tau":0.005,"buffer_size":100000,"batch_size":256,
            "normalization":"per_step"})

traj = run_episode(m, weather, ew=0.6, timestep_min=5)
rl_m = sim_metrics(traj)
bl_traj = run_baseline(weather, ew=0.6, timestep_min=5)
bl_m = sim_metrics(bl_traj)

print()
print("=" * 60)
print("  SAC 30k (fixed norm) vs Baseline")
print("=" * 60)
print(f"{'Metric':<25} {'Baseline':>12} {'RL Agent':>12} {'Delta':>12} {'':>5}")
print("-" * 60)
b = bl_m['comfort_score']; r = rl_m['comfort_score']
print(f"{'Comfort Score':<25} {b:>11.2f}% {r:>11.2f}% {r-b:>+11.2f}% {'WIN' if r>b else 'LOSE'}")
b = bl_m['total_energy']; r = rl_m['total_energy']
print(f"{'Energy (kWh/m2)':<25} {b:>12.2f} {r:>12.2f} {r-b:>+12.2f} {'WIN' if r<b else 'LOSE'}")
b = bl_m['total_discomfort']; r = rl_m['total_discomfort']
print(f"{'Discomfort (Kh)':<25} {b:>12.2f} {r:>12.2f} {r-b:>+12.2f} {'WIN' if r<b else 'LOSE'}")
b = bl_m['total_reward']; r = rl_m['total_reward']
print(f"{'Total Reward':<25} {b:>12.2f} {r:>12.2f} {r-b:>+12.2f} {'WIN' if r>b else 'LOSE'}")
bc = bl_m['cost_usd']; rc = rl_m['cost_usd']
print(f"{'Energy Cost':<25} ${bc:>11.2f} ${rc:>11.2f} ${rc-bc:>+11.2f} {'WIN' if rc<bc else 'LOSE'}")
print("=" * 60)
