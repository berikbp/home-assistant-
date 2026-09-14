#!/usr/bin/env python3
"""Evaluate best saved agents against baseline."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from rl_platform import load_weather, sim_metrics, run_episode, run_baseline, list_agents

weather = load_weather(5)
bl_traj = run_baseline(weather, ew=0.6, timestep_min=5)
bl_m = sim_metrics(bl_traj)

agents = list_agents()
for ag in agents:
    from rl_platform import load_agent_model
    model, meta = load_agent_model(ag["name"])
    if model is None:
        continue
    traj = run_episode(model, weather, ew=0.6, timestep_min=5)
    rl_m = sim_metrics(traj)
    n = meta["train_cfg"]["timesteps"]
    print(f"\n{'='*60}")
    print(f"  {meta['name']} ({meta['algo']}, {n} steps)")
    print(f"{'='*60}")
    print(f"{'Metric':<25} {'Baseline':>12} {'RL Agent':>12} {'Delta':>12}")
    print("-"*60)
    b = bl_m['comfort_score']; r = rl_m['comfort_score']
    print(f"{'Comfort Score':<25} {b:>11.2f}% {r:>11.2f}% {r-b:>+11.2f}%")
    b = bl_m['total_energy']; r = rl_m['total_energy']
    print(f"{'Energy (kWh/m2)':<25} {b:>12.2f} {r:>12.2f} {r-b:>+12.2f}")
    b = bl_m['total_discomfort']; r = rl_m['total_discomfort']
    print(f"{'Discomfort (Kh)':<25} {b:>12.2f} {r:>12.2f} {r-b:>+12.2f}")
    b = bl_m['total_reward']; r = rl_m['total_reward']
    print(f"{'Total Reward':<25} {b:>12.2f} {r:>12.2f} {r-b:>+12.2f}")
    bc = bl_m['cost_usd']; rc = rl_m['cost_usd']
    print(f"{'Energy Cost':<25} ${bc:>11.2f} ${rc:>11.2f} ${rc-bc:>+11.2f}")
    print("="*60)
