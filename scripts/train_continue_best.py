#!/usr/bin/env python3
"""Continue training best models and evaluate."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from rl_platform import (load_weather, SimpleHVACEnv, SAC, CB, State,
                          save_agent, sim_metrics, run_episode, run_baseline, load_agent_model)
from stable_baselines3.common.monitor import Monitor

weather = load_weather(5)
extra = 50000

for name in ["sac_20260911_154532", "sac_30k_fresh"]:
    model, meta = load_agent_model(name)
    if model is None:
        print(f"SKIP: {name} not found")
        continue

    prev_ts = meta["train_cfg"]["timesteps"]
    ew = meta["train_cfg"].get("energy_weight", 0.6)
    lr = meta["train_cfg"].get("lr", 3e-4)
    print(f"\n{'='*60}")
    print(f"  Continuing {name} ({prev_ts} steps -> {prev_ts+extra})")
    print(f"  ew={ew}, lr={lr}")
    print(f"{'='*60}")

    env = Monitor(SimpleHVACEnv(weather, ew, timestep_min=5))
    st = State()
    cb = CB(st)
    model.set_env(env)
    model.learn(total_timesteps=extra, callback=cb, reset_num_timesteps=False)
    env.close()

    rw = [float(x) for x in st.get("rewards")]
    cf = [float(x) for x in st.get("comforts")]
    en = [float(x) for x in st.get("energies")]
    metrics = {"episode_rewards": rw, "episode_comforts": cf, "episode_energies": en,
               "avg_reward": sum(rw)/len(rw) if rw else 0,
               "avg_comfort": sum(cf)/len(cf) if cf else 0,
               "avg_energy": sum(en)/len(en) if en else 0}
    total_ts = prev_ts + extra
    new_name = f"{name}_cont_{total_ts}"
    save_agent(new_name, meta["algo"], model, metrics,
               {**meta["train_cfg"], "timesteps": total_ts})
    print(f"Saved: {new_name}")

    # Evaluate
    traj = run_episode(model, weather, ew=0.6, timestep_min=5)
    rl_m = sim_metrics(traj)
    bl_traj = run_baseline(weather, ew=0.6, timestep_min=5)
    bl_m = sim_metrics(bl_traj)

    print(f"\n  {new_name} vs Baseline:")
    print(f"  {'Metric':<20} {'Baseline':>10} {'RL':>10} {'Delta':>10} {'':>5}")
    print(f"  {'-'*55}")
    b=bl_m['comfort_score']; r=rl_m['comfort_score']
    print(f"  {'Comfort':<20} {b:>9.2f}% {r:>9.2f}% {r-b:>+9.2f}% {'WIN' if r>b else 'LOSE'}")
    b=bl_m['total_energy']; r=rl_m['total_energy']
    print(f"  {'Energy':<20} {b:>10.2f} {r:>10.2f} {r-b:>+10.2f} {'WIN' if r<b else 'LOSE'}")
    b=bl_m['total_discomfort']; r=rl_m['total_discomfort']
    print(f"  {'Discomfort':<20} {b:>10.2f} {r:>10.2f} {r-b:>+10.2f} {'WIN' if r<b else 'LOSE'}")
    b=bl_m['total_reward']; r=rl_m['total_reward']
    print(f"  {'Reward':<20} {b:>10.2f} {r:>10.2f} {r-b:>+10.2f} {'WIN' if r>b else 'LOSE'}")
    bc=bl_m['cost_usd']; rc=rl_m['cost_usd']
    print(f"  {'Cost':<20} {'$'+f'{bc:.2f}':>10} {'$'+f'{rc:.2f}':>10} {'$'+f'{rc-bc:.2f}':>10} {'WIN' if rc<bc else 'LOSE'}")

print("\nDone.")
