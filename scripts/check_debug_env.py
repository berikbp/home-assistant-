from stable_baselines3.common.env_checker import (
    check_env,
)

from src.boptest_env import (
    BoptestDebugEnv,
)


def main():

    # ========================================================
    # 1. Validate Gymnasium/SB3 interface
    # ========================================================

    env = BoptestDebugEnv(
        max_steps=24
    )

    print(
        "Checking Gymnasium environment..."
    )

    check_env(
        env,
        warn=True,
    )

    print()
    print(
        "Environment check passed."
    )

    # ========================================================
    # 2. Fresh episode
    # ========================================================

    observation, info = (
        env.reset()
    )

    print()
    print(
        "Initial state"
    )

    print(
        "--------------------------------"
    )

    print(
        f"Simulation time: "
        f"{info['time']} s"
    )

    print(
        f"Environment step: "
        f"{info['current_step']}"
    )

    print(
        "Observation:"
    )

    print(
        observation
    )

    print()
    print(
        "Running 24 random actions..."
    )

    print()

    # ========================================================
    # 3. Exactly 24 agent actions
    # ========================================================

    for i in range(24):

        action = (
            env.action_space.sample()
        )

        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(
            action
        )

        print(
            f"Step "
            f"{info['current_step']:02d} | "
            f"time="
            f"{info['time']/3600:5.1f} h | "
            f"RL action="
            f"{info['agent_action']:+.3f} | "
            f"fan="
            f"{info['fan_command']:.3f} | "
            f"room="
            f"{info['room_temperature_C']:.2f} °C | "
            f"fan power="
            f"{info['fan_power_W']:.2f} W"
        )

        if (
            terminated
            or truncated
        ):

            print()
            print(
                "Episode ended."
            )

            print(
                f"Agent actions executed: "
                f"{info['current_step']}"
            )

            print(
                f"Final simulation time: "
                f"{info['time']/3600:.1f} h"
            )

            break

    env.close()

    print()
    print(
        "Environment closed."
    )


if __name__ == "__main__":
    main()