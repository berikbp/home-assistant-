import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from stable_baselines3 import SAC

from src.boptest_rl_env import BoptestRLEnv


# ============================================================
# Configuration
# ============================================================

BASE_URL = "http://127.0.0.1:8000"
TESTCASE = "bestest_air"

ENERGY_WEIGHT = 0.6
EPISODE_STEPS = 336  # 14 days

MODEL_DIR = Path("checkpoints/sac")
OUTPUT_DIR = Path("data/sac_eval_14d")


# ============================================================
# Load model
# ============================================================

def load_model():
    best_path = MODEL_DIR / "best" / "best_model"
    final_path = MODEL_DIR / "final"

    if best_path.exists():
        print(f"Loading best model: {best_path}")
        return SAC.load(str(best_path))

    if final_path.exists():
        print(f"Loading final model: {final_path}")
        return SAC.load(str(final_path))

    raise FileNotFoundError(
        f"No model found in {MODEL_DIR}. "
        "Run train_sac.py first."
    )


# ============================================================
# Evaluate
# ============================================================

def evaluate(model, env, n_episodes=1):
    """Run n full episodes and collect trajectories."""

    all_results = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0
        step = 0
        trajectory = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            step += 1

            trajectory.append({
                "step": step,
                "time": info["time"],
                "zone_temperature_c": info["zone_temperature_c"],
                "lower_setpoint_c": info["lower_setpoint_c"],
                "upper_setpoint_c": info["upper_setpoint_c"],
                "occupancy": info.get("occupancy", 0),
                "fan_command": info["fan_command"],
                "supply_temperature_k": info["supply_temperature_k"],
                "power_heat_w": info["power_heat_w"],
                "power_cool_w": info["power_cool_w"],
                "power_fan_w": info["power_fan_w"],
                "discomfort_proxy": info["discomfort_proxy"],
                "energy_proxy_kwh_m2": info["energy_proxy_kwh_m2"],
                "reward": reward,
            })

        all_results.append({
            "episode": ep + 1,
            "total_reward": ep_reward,
            "steps": step,
        })

        print(
            f"Episode {ep + 1}: "
            f"reward={ep_reward:.3f} | "
            f"discomfort={sum(t['discomfort_proxy'] for t in trajectory):.3f} | "
            f"energy={sum(t['energy_proxy_kwh_m2'] for t in trajectory):.5f}"
        )

    return all_results, trajectory


# ============================================================
# Compute metrics (same as baseline scripts)
# ============================================================

def compute_metrics(trajectory_df):
    """Compute comfort and energy metrics matching baseline style."""

    room = trajectory_df["zone_temperature_c"].values
    lower = trajectory_df["lower_setpoint_c"].values
    upper = trajectory_df["upper_setpoint_c"].values
    occupancy = trajectory_df["occupancy"].values

    # Comfort
    comfortable = ((room >= lower) & (room <= upper)).astype(int)
    occupied = (occupancy > 0).astype(int)

    comfort_24h = 100.0 * comfortable.mean()

    occ_mask = occupied == 1
    if occ_mask.sum() > 0:
        comfort_occupied = 100.0 * comfortable[occ_mask].mean()
    else:
        comfort_occupied = None

    # Discomfort (K·h)
    lower_violation = np.maximum(lower - room, 0.0)
    upper_violation = np.maximum(room - upper, 0.0)
    discomfort_k = lower_violation + upper_violation

    step_hours = 3600.0 / 3600.0  # 1 hour per step
    discomfort_24h_kh = discomfort_k.sum() * step_hours
    discomfort_occupied_kh = (
        discomfort_k[occ_mask].sum() * step_hours
        if occ_mask.sum() > 0 else 0.0
    )

    # Energy (kWh/m²)
    floor_area = 48.0
    total_power_w = (
        trajectory_df["power_heat_w"].values
        + trajectory_df["power_cool_w"].values
        + trajectory_df["power_fan_w"].values
    )
    energy_kwh_m2 = (
        total_power_w / 1000.0 * step_hours / floor_area
    )

    return {
        "comfort_score_24h_percent": comfort_24h,
        "comfort_score_occupied_percent": comfort_occupied,
        "sampled_discomfort_24h_Kh": discomfort_24h_kh,
        "sampled_discomfort_occupied_Kh": discomfort_occupied_kh,
        "sampled_hvac_energy_kWh_m2": energy_kwh_m2.sum(),
        "total_energy_kWh_m2": energy_kwh_m2.sum(),
    }


# ============================================================
# Main
# ============================================================

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------
    # 1. Load model
    # ----------------------------------------------------------

    model = load_model()

    # ----------------------------------------------------------
    # 2. Create 14-day evaluation environment
    # ----------------------------------------------------------

    print()
    print("Creating 14-day evaluation environment...")

    env = BoptestRLEnv(
        base_url=BASE_URL,
        testcase=TESTCASE,
        energy_weight=ENERGY_WEIGHT,
        episode_steps=EPISODE_STEPS,
    )

    # ----------------------------------------------------------
    # 3. Evaluate
    # ----------------------------------------------------------

    print()
    print("Running 14-day evaluation...")

    results, trajectory = evaluate(model, env, n_episodes=1)

    # ----------------------------------------------------------
    # 4. Compute metrics
    # ----------------------------------------------------------

    traj_df = pd.DataFrame(trajectory)
    metrics = compute_metrics(traj_df)

    # ----------------------------------------------------------
    # 5. Load baseline for comparison
    # ----------------------------------------------------------

    baseline_path = Path("data/baseline_14d/summary_corrected.json")

    if baseline_path.exists():
        with open(baseline_path) as f:
            baseline = json.load(f)
    else:
        baseline = None

    # ----------------------------------------------------------
    # 6. Save results
    # ----------------------------------------------------------

    traj_df.to_csv(OUTPUT_DIR / "trajectory.csv", index=False)

    sac_summary = {
        "method": "SAC",
        "energy_weight": ENERGY_WEIGHT,
        "episode_steps": EPISODE_STEPS,
        **metrics,
    }

    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(sac_summary, f, indent=2)

    # ----------------------------------------------------------
    # 7. Print comparison
    # ----------------------------------------------------------

    print()
    print("=" * 60)
    print("SAC vs BASELINE (14-day)")
    print("=" * 60)

    header = f"{'Metric':<40s} {'Baseline':>12s} {'SAC':>12s}"
    print(header)
    print("-" * 60)

    comparisons = [
        (
            "Comfort Score 24/7 (%)",
            baseline.get("comfort_score_24h_percent") if baseline else None,
            metrics["comfort_score_24h_percent"],
        ),
        (
            "Comfort Score Occupied (%)",
            baseline.get("comfort_score_occupied_percent") if baseline else None,
            metrics["comfort_score_occupied_percent"],
        ),
        (
            "Discomfort 24/7 (K·h)",
            baseline.get("sampled_discomfort_24h_Kh") if baseline else None,
            metrics["sampled_discomfort_24h_Kh"],
        ),
        (
            "Discomfort Occupied (K·h)",
            baseline.get("sampled_discomfort_occupied_Kh") if baseline else None,
            metrics["sampled_discomfort_occupied_Kh"],
        ),
        (
            "Energy (kWh/m²)",
            baseline.get("sampled_hvac_energy_total_kWh", 0) / 48.0
            if baseline else None,
            metrics["total_energy_kWh_m2"],
        ),
    ]

    for name, bl_val, sac_val in comparisons:
        bl_str = f"{bl_val:.2f}" if bl_val is not None else "N/A"
        sac_str = f"{sac_val:.2f}" if sac_val is not None else "N/A"

        if bl_val is not None and sac_val is not None:
            if "Comfort" in name:
                delta = sac_val - bl_val
                arrow = "+" if delta > 0 else ""
                note = f"({arrow}{delta:.1f})"
            else:
                delta = sac_val - bl_val
                arrow = "+" if delta > 0 else ""
                note = f"({arrow}{delta:.2f})"
        else:
            note = ""

        print(f"{name:<40s} {bl_str:>12s} {sac_str:>12s} {note}")

    print()
    print(f"Results saved to: {OUTPUT_DIR}")

    # ----------------------------------------------------------
    # 8. Cleanup
    # ----------------------------------------------------------

    env.close()


if __name__ == "__main__":
    main()
