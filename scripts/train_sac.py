import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor

from src.boptest_rl_env import BoptestRLEnv


# ============================================================
# Configuration
# ============================================================

BASE_URL = "http://127.0.0.1:8000"
TESTCASE = "bestest_air"

ENERGY_WEIGHT = 0.6
EPISODE_STEPS = 24

TOTAL_TIMESTEPS = 50_000
EVAL_FREQ_STEPS = 10 * EPISODE_STEPS
N_EVAL_EPISODES = 3

LOG_DIR = Path("logs/sac")
CHECKPOINT_DIR = Path("checkpoints/sac")


# ============================================================
# Callbacks
# ============================================================

class EpisodeInfoCallback(BaseCallback):
    """Print per-episode reward and key metrics."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_comforts = []
        self.episode_energies = []
        self._current_reward = 0.0
        self._current_discomfort = 0.0
        self._current_energy = 0.0
        self._steps = 0

    def _on_step(self) -> bool:
        self._current_reward += self.locals["rewards"][0]
        self._current_discomfort += self.locals["infos"][0].get(
            "discomfort_proxy", 0.0
        )
        self._current_energy += self.locals["infos"][0].get(
            "energy_proxy_kwh_m2", 0.0
        )
        self._steps += 1

        if self.locals["dones"][0]:
            self.episode_rewards.append(self._current_reward)
            self.episode_comforts.append(self._current_discomfort)
            self.episode_energies.append(self._current_energy)

            n = len(self.episode_rewards)
            avg_r = np.mean(self.episode_rewards[-50:])
            avg_d = np.mean(self.episode_comforts[-50:])
            avg_e = np.mean(self.episode_energies[-50:])

            if n % 10 == 0 or n <= 5:
                print(
                    f"Episode {n:4d} | "
                    f"avg_reward(50)={avg_r:8.3f} | "
                    f"avg_discomfort={avg_d:8.3f} | "
                    f"avg_energy={avg_e:8.5f}"
                )

            self._current_reward = 0.0
            self._current_discomfort = 0.0
            self._current_energy = 0.0
            self._steps = 0

        return True


# ============================================================
# Evaluation environment (separate BOPTEST instance)
# ============================================================

def make_eval_env():
    env = BoptestRLEnv(
        base_url=BASE_URL,
        testcase=TESTCASE,
        energy_weight=ENERGY_WEIGHT,
        episode_steps=EPISODE_STEPS,
    )
    env = Monitor(env)
    return env


# ============================================================
# Main
# ============================================================

def main():

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------
    # 1. Create training environment
    # ----------------------------------------------------------

    print("Creating training environment...")

    train_env = BoptestRLEnv(
        base_url=BASE_URL,
        testcase=TESTCASE,
        energy_weight=ENERGY_WEIGHT,
        episode_steps=EPISODE_STEPS,
    )

    train_env = Monitor(train_env)

    print(f"Observation space: {train_env.observation_space}")
    print(f"Action space: {train_env.action_space}")

    # ----------------------------------------------------------
    # 2. SAC hyperparameters
    # ----------------------------------------------------------

    model = SAC(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        buffer_size=50_000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        ent_coef="auto",
        target_entropy="auto",
        verbose=0,
        tensorboard_log=None,
    )

    print()
    print("SAC model created.")
    print(f"Policy: {model.policy}")
    print(f"Total timesteps: {TOTAL_TIMESTEPS:,}")

    # ----------------------------------------------------------
    # 3. Callbacks
    # ----------------------------------------------------------

    info_cb = EpisodeInfoCallback()

    eval_env = make_eval_env()

    eval_cb = EvalCallback(
        eval_env,
        n_eval_episodes=N_EVAL_EPISODES,
        eval_freq=EVAL_FREQ_STEPS,
        best_model_save_path=str(CHECKPOINT_DIR / "best"),
        log_path=str(LOG_DIR / "eval"),
        verbose=1,
    )

    # ----------------------------------------------------------
    # 4. Train
    # ----------------------------------------------------------

    print()
    print("Starting training...")
    print("=" * 60)

    wall_start = time.perf_counter()

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=[info_cb, eval_cb],
            progress_bar=False,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")

    wall_elapsed = time.perf_counter() - wall_start

    # ----------------------------------------------------------
    # 5. Save final model
    # ----------------------------------------------------------

    final_path = CHECKPOINT_DIR / "final"
    model.save(str(final_path))

    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Wall time: {wall_elapsed:.1f} s")
    print(f"Episodes completed: {len(info_cb.episode_rewards)}")
    print(f"Final model saved: {final_path}")

    if info_cb.episode_rewards:
        print(f"Last 50 avg reward: {np.mean(info_cb.episode_rewards[-50:]):.3f}")
        print(f"Last 50 avg discomfort: {np.mean(info_cb.episode_comforts[-50:]):.3f}")
        print(f"Last 50 avg energy: {np.mean(info_cb.episode_energies[-50:]):.5f}")

    # ----------------------------------------------------------
    # 6. Save training summary
    # ----------------------------------------------------------

    summary = {
        "algorithm": "SAC",
        "energy_weight": ENERGY_WEIGHT,
        "episode_steps": EPISODE_STEPS,
        "total_timesteps": TOTAL_TIMESTEPS,
        "episodes_completed": len(info_cb.episode_rewards),
        "wall_runtime_seconds": wall_elapsed,
        "final_avg_reward_50": (
            float(np.mean(info_cb.episode_rewards[-50:]))
            if info_cb.episode_rewards else None
        ),
        "final_avg_discomfort_50": (
            float(np.mean(info_cb.episode_comforts[-50:]))
            if info_cb.episode_comforts else None
        ),
        "final_avg_energy_50": (
            float(np.mean(info_cb.episode_energies[-50:]))
            if info_cb.episode_energies else None
        ),
    }

    with open(CHECKPOINT_DIR / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ----------------------------------------------------------
    # 7. Cleanup
    # ----------------------------------------------------------

    train_env.close()


if __name__ == "__main__":
    main()
