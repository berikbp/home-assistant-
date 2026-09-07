import numpy as np

from src.boptest_rl_env import BoptestRLEnv


def main():

    env = BoptestRLEnv(
        energy_weight=0.6,
        episode_steps=24,
    )

    try:
        obs, info = env.reset()

        print()
        print("Initial observation:")
        print(obs)

        print()
        print("Initial info:")
        print(info)

        total_reward = 0.0

        # Neutral normalized action:
        #
        # fan = 0.5
        # supply temperature = 26 C
        action = np.array(
            [0.0, 0.0],
            dtype=np.float32,
        )

        for step in range(24):

            (
                obs,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(action)

            total_reward += reward

            print(
                f"{step + 1:02d} "
                f"time={info['time']:8.0f} "
                f"Tz={info['zone_temperature_c']:6.2f} C "
                f"fan={info['fan_command']:.3f} "
                f"Tsup={info['supply_temperature_k'] - 273.15:6.2f} C "
                f"D={info['discomfort_proxy']:7.3f} "
                f"E={info['energy_proxy_kwh_m2']:8.5f} "
                f"reward={reward:9.3f}"
            )

            if terminated or truncated:
                break

        print()
        print("Episode complete.")
        print(f"Total reward: {total_reward:.3f}")
        print(f"Final simulated time: {info['time']:.0f} s")

    finally:
        env.close()


if __name__ == "__main__":
    main()
