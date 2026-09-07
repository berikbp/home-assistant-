import requests
import numpy as np


BASE_URL = "http://127.0.0.1:8000"


def main():

    # --------------------------------------------------------
    # 1. Select test case
    # --------------------------------------------------------

    response = requests.post(
        f"{BASE_URL}/testcases/bestest_air/select",
        timeout=60,
    )

    response.raise_for_status()

    testid = response.json()["testid"]

    print(f"Test ID: {testid}")

    try:

        # ----------------------------------------------------
        # 2. Initialize at t = 0
        # ----------------------------------------------------

        response = requests.put(
            f"{BASE_URL}/initialize/{testid}",
            json={
                "start_time": 0,
                "warmup_period": 0,
            },
            timeout=60,
        )

        response.raise_for_status()

        print("Initialized.")

        # ----------------------------------------------------
        # 3. Advance ONE controller step = 3600 s
        #
        # No override → baseline controller.
        # ----------------------------------------------------

        response = requests.post(
            f"{BASE_URL}/advance/{testid}",
            json={},
            timeout=60,
        )

        response.raise_for_status()

        data = response.json()["payload"]

        print(
            f"Advance finished at "
            f"{data['time']} s"
        )

        # ----------------------------------------------------
        # 4. Ask BOPTEST for stored trajectory
        # ----------------------------------------------------

        response = requests.put(
            f"{BASE_URL}/results/{testid}",
            json={
                "point_names": [
                    "zon_reaTRooAir_y",
                    "fcu_reaPFan_y",
                    "fcu_reaPHea_y",
                    "fcu_reaPCoo_y",
                ],
                "start_time": 0,
                "final_time": 3600,
            },
            timeout=60,
        )

        response.raise_for_status()

        result = response.json()

        if result.get("status") != 200:
            raise RuntimeError(result)

        payload = result["payload"]

        times = np.asarray(
            payload["time"],
            dtype=float,
        )

        print()
        print("========================================")
        print("STORED RESULT RESOLUTION")
        print("========================================")

        print(
            f"Number of stored points: "
            f"{len(times)}"
        )

        if len(times) > 0:

            print(
                f"First timestamp: "
                f"{times[0]} s"
            )

            print(
                f"Last timestamp: "
                f"{times[-1]} s"
            )

        if len(times) > 1:

            dt = np.diff(times)

            print(
                f"Minimum spacing: "
                f"{dt.min():.6f} s"
            )

            print(
                f"Maximum spacing: "
                f"{dt.max():.6f} s"
            )

            print(
                f"Median spacing: "
                f"{np.median(dt):.6f} s"
            )

            unique_dt = np.unique(
                np.round(dt, 6)
            )

            print()
            print(
                "Unique spacings "
                "(first 20):"
            )

            print(
                unique_dt[:20]
            )

        # ----------------------------------------------------
        # 5. Print first few stored states
        # ----------------------------------------------------

        print()
        print("First stored points")
        print("----------------------------------------")

        n_show = min(
            15,
            len(times),
        )

        for i in range(n_show):

            room_C = (
                payload[
                    "zon_reaTRooAir_y"
                ][i]
                - 273.15
            )

            print(
                f"t={times[i]:8.2f} s | "
                f"room={room_C:7.3f} °C | "
                f"fan={payload['fcu_reaPFan_y'][i]:8.3f} W | "
                f"heat={payload['fcu_reaPHea_y'][i]:8.3f} W | "
                f"cool={payload['fcu_reaPCoo_y'][i]:8.3f} W"
            )

    finally:

        requests.put(
            f"{BASE_URL}/stop/{testid}",
            timeout=30,
        )

        print()
        print("Test stopped.")


if __name__ == "__main__":
    main()
