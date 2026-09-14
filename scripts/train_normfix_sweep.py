#!/usr/bin/env python3
"""SAC 30k fixed norm + ew=0.7 for better balance."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from rl_platform import (load_weather, SimpleHVACEnv, SAC, CB, State,
                          save_agent, sim_metrics, run_episode, run_baseline)
from stable_baselines3.common.monitor import Monitor

weather = load_weather(5)

for ew in [0.5, 0.7]:
    env = Monitor(SimpleHVACEnv(weather, ew, timestep_min=5))
    st = State()
    cb = CB(st)
    print(f"Training SAC 30k steps (ew={ew})...")
    m = SAC("MlpPolicy", env, learning_rate=3e-4, buffer_size=100000,
            batch_size=256, gamma=0.99, tau=0.005, verbose=0)
    m.learn(total_timesteps=30000, callback=cb)
    env.close()

    rw = [float(x) for x in st.get("rewards")]
    cf = [float(x) for x in st.get("comforts")]
    en = [float(x) for x in st.get("energies")]
    metrics = {"episode_rewards": rw, "episode_comforts": cf, "episode_energies": en,
               "avg_reward": sum(rw)/len(rw) if rw else 0,
               "avg_comfort": sum(cf)/len(cf) if cf else 0,
               "avg_energy": sum(en)/len(en) if en else 0}
    save_agent(f"sac_30k_ew{int(ew*10)}_normfix", "SAC", m, metrics,
               {"algo":"SAC","timesteps":30000,"energy_weight":ew,"lr":3e-4,
                "timestep_min":5,"gamma":0.99,"tau":0.005,"buffer_size":100000,"batch_size":256})

    # Evaluate with ew=0.6 (standard comparison)
    traj = run_episode(m, weather, ew=0.6, timestep_min=5)
    rl_m = sim_metrics(traj)
    bl_traj = run_baseline(weather, ew=0.6, timestep_min=5)
    bl_m = sim_metrics(bl_traj)

    wins = 0
    print(f"\n  ew={ew} vs Baseline:")
    b = bl_m['comfort_score']; r = rl_m['comfort_score']
    w = "WIN" if r>b else "LOSE"; wins += (1 if "WIN" in w else 0)
    print(f"  Comfort:  {b:.2f}% -> {r:.2f}% ({r-b:+.2f}%) {w}")
    b = bl_m['total_energy']; r = rl_m['total_energy']
    w = "WIN" if r<b else "LOSE"; wins += (1 if "WIN" in w else 0)
    print(f"  Energy:   {b:.2f} -> {r:.2f} ({r-b:+.2f}) {w}")
    b = bl_m['total_discomfort']; r = rl_m['total_discomfort']
    w = "WIN" if r<b else "LOSE"; wins += (1 if "WIN" in w else 0)
    print(f"  Discomf:  {b:.2f} -> {r:.2f} ({r-b:+.2f}) {w}")
    b = bl_m['total_reward']; r = rl_m['total_reward']
    w = "WIN" if r>b else "LOSE"; wins += (1 if "WIN" in w else 0)
    print(f"  Reward:   {b:.2f} -> {r:.2f} ({r-b:+.2f}) {w}")
    bc = bl_m['cost_usd']; rc = rl_m['cost_usd']
    w = "WIN" if rc<bc else "LOSE"; wins += (1 if "WIN" in w else 0)
    print(f"  Cost:     ${bc:.2f} -> ${rc:.2f} (${rc-bc:+.2f}) {w}")
    print(f"  Score: {wins}/5")
