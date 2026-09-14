#!/usr/bin/env python3
"""Train 2 models to 100k steps, evaluate at 30k/60k/100k checkpoints."""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))

from rl_platform import (load_weather, SimpleHVACEnv, SAC, DDPG, CB, State,
                          save_agent, sim_metrics, run_episode, run_baseline)
from stable_baselines3.common.monitor import Monitor

weather = load_weather(5)
bl_traj = run_baseline(weather, ew=0.6, timestep_min=5)
bl_m = sim_metrics(bl_traj)
print(f"Baseline: comfort={bl_m['comfort_score']:.2f}% energy={bl_m['total_energy']:.2f} "
      f"discomfort={bl_m['total_discomfort']:.2f} reward={bl_m['total_reward']:.2f} "
      f"cost=${bl_m['cost_usd']:.2f}")

results = {}

for algo_name, cls in [("SAC", SAC), ("DDPG", DDPG)]:
    print(f"\n{'='*60}")
    print(f"  Training {algo_name}")
    print(f"{'='*60}")

    env = Monitor(SimpleHVACEnv(weather, 0.6, timestep_min=5))
    st = State()
    cb = CB(st)

    extra_kw = {}
    if algo_name in ("SAC", "DDPG"):
        extra_kw = {"buffer_size": 100000, "tau": 0.005}

    m = cls("MlpPolicy", env, learning_rate=3e-4, batch_size=256,
            gamma=0.99, verbose=1, **extra_kw)

    checkpoints = [30000, 60000, 100000]
    prev = 0
    for target in checkpoints:
        steps = target - prev
        print(f"\n--- {algo_name}: training to {target} steps ---")
        m.learn(total_timesteps=steps, callback=cb, reset_num_timesteps=(prev == 0))
        prev = target

        # Evaluate
        env_eval = SimpleHVACEnv(weather, 0.6, timestep_min=5)
        traj = run_episode(m, weather, ew=0.6, timestep_min=5)
        rl_m = sim_metrics(traj)

        name = f"{algo_name.lower()}_{target}k_orig"
        save_agent(name, algo_name, m,
                   {"episode_rewards": [float(x) for x in st.get("rewards")],
                    "episode_comforts": [float(x) for x in st.get("comforts")],
                    "episode_energies": [float(x) for x in st.get("energies")],
                    "avg_reward": st.get("rewards")[-1] if st.get("rewards") else 0,
                    "avg_comfort": st.get("comforts")[-1] if st.get("comforts") else 0,
                    "avg_energy": st.get("energies")[-1] if st.get("energies") else 0},
                   {"algo": algo_name, "timesteps": target, "energy_weight": 0.6,
                    "lr": 3e-4, "timestep_min": 5, "gamma": 0.99, "tau": 0.005,
                    "buffer_size": 100000, "batch_size": 256})

        wins = 0
        print(f"\n  {name} at {target}k vs Baseline:")
        b = bl_m['comfort_score']; r = rl_m['comfort_score']
        w = "WIN" if r > b else "LOSE"; wins += (1 if "WIN" in w else 0)
        print(f"    Comfort:  {b:.2f}% -> {r:.2f}% ({r-b:+.2f}%) {w}")
        b = bl_m['total_energy']; r = rl_m['total_energy']
        w = "WIN" if r < b else "LOSE"; wins += (1 if "WIN" in w else 0)
        print(f"    Energy:   {b:.2f} -> {r:.2f} ({r-b:+.2f}) {w}")
        b = bl_m['total_discomfort']; r = rl_m['total_discomfort']
        w = "WIN" if r < b else "LOSE"; wins += (1 if "WIN" in w else 0)
        print(f"    Discomf:  {b:.2f} -> {r:.2f} ({r-b:+.2f}) {w}")
        b = bl_m['total_reward']; r = rl_m['total_reward']
        w = "WIN" if r > b else "LOSE"; wins += (1 if "WIN" in w else 0)
        print(f"    Reward:   {b:.2f} -> {r:.2f} ({r-b:+.2f}) {w}")
        bc = bl_m['cost_usd']; rc = rl_m['cost_usd']
        w = "WIN" if rc < bc else "LOSE"; wins += (1 if "WIN" in w else 0)
        print(f"    Cost:     ${bc:.2f} -> ${rc:.2f} (${rc-bc:+.2f}) {w}")
        print(f"    Score: {wins}/5")

        if algo_name not in results:
            results[algo_name] = []
        results[algo_name].append({
            "steps": target, "wins": wins,
            "comfort_delta": rl_m['comfort_score'] - bl_m['comfort_score'],
            "energy_delta": rl_m['total_energy'] - bl_m['total_energy'],
            "discomfort_delta": rl_m['total_discomfort'] - bl_m['total_discomfort'],
            "reward_delta": rl_m['total_reward'] - bl_m['total_reward'],
        })

    env.close()

print("\n\n" + "=" * 60)
print("  LEARNING CURVES")
print("=" * 60)
for algo, data in results.items():
    print(f"\n  {algo}:")
    for d in data:
        print(f"    {d['steps']:>5}k: comfort={d['comfort_delta']:+.2f}% "
              f"energy={d['energy_delta']:+.2f} discomfort={d['discomfort_delta']:+.2f} "
              f"reward={d['reward_delta']:+.2f} score={d['wins']}/5")

Path = __import__('pathlib').Path
(Path(__file__).parent.parent / "models" / "learning_curves.json").write_text(
    json.dumps(results, indent=2))
print("\nSaved learning_curves.json")
