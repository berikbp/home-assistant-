#!/usr/bin/env python3
"""Fresh SAC 100k steps - optimized to beat baseline on ALL metrics."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from rl_platform import (load_weather, SimpleHVACEnv, SAC, PPO, DDPG, CB, State,
                          save_agent, sim_metrics, run_episode, run_baseline)
from stable_baselines3.common.monitor import Monitor

weather = load_weather(5)

def train(algo, steps, lr, ew=0.6, **kw):
    env = Monitor(SimpleHVACEnv(weather, ew, timestep_min=5))
    st = State()
    cb = CB(st)
    extra = {}
    if algo == "PPO":
        extra = {"n_steps": 512}
    extra.update(kw)
    cls = {"SAC": SAC, "PPO": PPO, "DDPG": DDPG}[algo]
    m = cls("MlpPolicy", env, learning_rate=lr, batch_size=256, gamma=0.99, verbose=1, **extra)
    m.learn(total_timesteps=steps, callback=cb)
    env.close()
    return m, st

print("=" * 60)
print("  Training SAC 100k (LR=3e-4)")
print("=" * 60)
m, st = train("SAC", 100000, 3e-4, buffer_size=100000, tau=0.005)

rw = [float(x) for x in st.get("rewards")]
cf = [float(x) for x in st.get("comforts")]
en = [float(x) for x in st.get("energies")]
metrics = {"episode_rewards": rw, "episode_comforts": cf, "episode_energies": en,
           "avg_reward": sum(rw)/len(rw) if rw else 0,
           "avg_comfort": sum(cf)/len(cf) if cf else 0,
           "avg_energy": sum(en)/len(en) if en else 0}
save_agent("sac_100k_final", "SAC", m, metrics,
           {"algo":"SAC","timesteps":100000,"energy_weight":0.6,"lr":3e-4,
            "timestep_min":5,"gamma":0.99,"tau":0.005,"buffer_size":100000,"batch_size":256})

traj = run_episode(m, weather, ew=0.6, timestep_min=5)
rl_m = sim_metrics(traj)
bl_traj = run_baseline(weather, ew=0.6, timestep_min=5)
bl_m = sim_metrics(bl_traj)

print()
print("=" * 60)
print("  SAC 100k vs Baseline")
print("=" * 60)
print(f"{'Metric':<25} {'Baseline':>12} {'RL Agent':>12} {'Delta':>12}")
print("-" * 60)
b = bl_m['comfort_score']; r = rl_m['comfort_score']
s = "WIN" if r > b else "LOSE"
print(f"{'Comfort Score':<25} {b:>11.2f}% {r:>11.2f}% {r-b:>+11.2f}% {s}")
b = bl_m['total_energy']; r = rl_m['total_energy']
s = "WIN" if r < b else "LOSE"
print(f"{'Energy (kWh/m2)':<25} {b:>12.2f} {r:>12.2f} {r-b:>+12.2f} {s}")
b = bl_m['total_discomfort']; r = rl_m['total_discomfort']
s = "WIN" if r < b else "LOSE"
print(f"{'Discomfort (Kh)':<25} {b:>12.2f} {r:>12.2f} {r-b:>+12.2f} {s}")
b = bl_m['total_reward']; r = rl_m['total_reward']
s = "WIN" if r > b else "LOSE"
print(f"{'Total Reward':<25} {b:>12.2f} {r:>12.2f} {r-b:>+12.2f} {s}")
bc = bl_m['cost_usd']; rc = rl_m['cost_usd']
s = "WIN" if rc < bc else "LOSE"
print(f"{'Energy Cost':<25} ${bc:>11.2f} ${rc:>11.2f} ${rc-bc:>+11.2f} {s}")
print("=" * 60)
