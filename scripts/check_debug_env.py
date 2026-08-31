from stable_baselines3.common.env_checker import check_env

from src.boptest_env import BoptestDebugEnv


def main():

    env = BoptestDebugEnv()

    print("Checking Gymnasium environment...")

    check_env(
        env,
        warn=True,
    )

    print()
    print("Environment check passed.")

    obs, info = env.reset()

    print()
    print("Initial observation:")
    print(obs)

    print()
    print("Running 10 random actions...")
    print()

    for i in range(10):

        action = env.action_space.sample()

        (
            obs,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        print(
            f"Step {i + 1:02d} | "
            f"action={action[0]:.3f} | "
            f"room={info['room_temperature_C']:.2f} °C | "
            f"fan power={info['fan_power_W']:.2f} W | "
            f"reward={reward:.2f}"
        )

        if terminated or truncated:
            break

    env.close()

    print()
    print("Finished.")


if __name__ == "__main__":
    main()
