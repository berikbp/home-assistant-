#!/usr/bin/env python3
"""Diagnose reward scaling."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from rl_platform import load_weather, run_baseline, sim_metrics, SimpleHVACEnv

weather = load_weather(5)
bl = run_baseline(weather, ew=0.6, timestep_min=5)
m = sim_metrics(bl)
env = SimpleHVACEnv(weather, 0.6, timestep_min=5)
print(f"total_steps: {len(weather)}")
print(f"dref: {env.dref:.6f}")
print(f"eref: {env.eref:.6f}")
print(f"1/dref (penalty for 1 deg violation): {1.0/env.dref:.1f}")
print(f"1/eref (penalty for 1 kWh/m2): {1.0/env.eref:.1f}")
print(f"Baseline: comfort={m['comfort_score']:.2f} discomfort={m['total_discomfort']:.2f} energy={m['total_energy']:.4f} reward={m['total_reward']:.2f}")
print(f"Avg discomfort/step: {m['total_discomfort']/len(bl):.4f}")
print(f"Avg energy/step: {m['total_energy']/len(bl):.6f}")

# Calculate per-step reward components for PI baseline
dis_sum = 0
en_sum = 0
for t in bl:
    dis = t['discomfort_proxy']
    en = t['energy_proxy_kwh_m2']
    dn = dis / env.dref
    en_norm = en / env.eref
    dis_sum += dn
    en_sum += en_norm
print(f"\nPI baseline total dn (discomfort in reward space): {dis_sum:.2f}")
print(f"PI baseline total en*ew (energy in reward space): {en_sum * 0.6:.2f}")
print(f"PI baseline reward: -(dn + ew*en) = {-(dis_sum + 0.6*en_sum):.2f}")
