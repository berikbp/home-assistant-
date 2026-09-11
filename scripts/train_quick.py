#!/usr/bin/env python3
"""Quick training script: SAC 20k timesteps."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from rl_platform import load_weather, SimpleHVACEnv, SAC, CB, State, save_agent, sim_metrics, run_episode, run_baseline
from stable_baselines3.common.monitor import Monitor

weather = load_weather(5)
env = Monitor(SimpleHVACEnv(weather, 0.6, timestep_min=5))
st = State()
cb = CB(st)
print("Training SAC 20k timesteps...")
m = SAC("MlpPolicy", env, learning_rate=3e-4, buffer_size=50000, batch_size=256, gamma=0.99, tau=0.005, verbose=1)
m.learn(total_timesteps=20000, callback=cb)
env.close()
print("Training done. Saving...")

rw = [float(x) for x in st.get("rewards")]
cf = [float(x) for x in st.get("comforts")]
en = [float(x) for x in st.get("energies")]
metrics = {
    "episode_rewards": rw, "episode_comforts": cf, "episode_energies": en,
    "avg_reward": sum(rw)/len(rw) if rw else 0,
    "avg_comfort": sum(cf)/len(cf) if cf else 0,
    "avg_energy": sum(en)/len(en) if en else 0,
}
train_cfg = {"algo": "SAC", "timesteps": 20000, "energy_weight": 0.6, "lr": 3e-4,
             "timestep_min": 5, "gamma": 0.99, "tau": 0.005, "buffer_size": 50000, "batch_size": 256}
meta = save_agent("sac_20k_quick", "SAC", m, metrics, train_cfg)
print(f"Saved: {meta['name']}")

traj = run_episode(m, weather, ew=0.6, timestep_min=5)
mets = sim_metrics(traj)
print(f"Simulation: comfort={mets['comfort_score']:.1f}% energy={mets['total_energy']:.4f} kWh/m2 cost=${mets['cost_usd']:.4f}")
print("Done.")
