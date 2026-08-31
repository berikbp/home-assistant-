import requests


BASE_URL = "http://127.0.0.1:8000"


def main():

    # --------------------------------------------------------
    # 1. Select bestest_air
    # --------------------------------------------------------

    response = requests.post(
        f"{BASE_URL}/testcases/bestest_air/select"
    )

    response.raise_for_status()

    testid = response.json()["testid"]

    print(f"Test ID: {testid}")

    try:

        # ----------------------------------------------------
        # 2. Initialize
        # ----------------------------------------------------

        response = requests.put(
            f"{BASE_URL}/initialize/{testid}",
            json={
                "start_time": 0,
                "warmup_period": 0,
            },
        )

        response.raise_for_status()

        print("Initialized.")

        # ----------------------------------------------------
        # 3. Advance 8 hours using baseline controller
        # ----------------------------------------------------

        print()
        print("BASELINE CONTROL")
        print("--------------------------------")

        for step in range(8):

            response = requests.post(
                f"{BASE_URL}/advance/{testid}",
                json={},
            )

            response.raise_for_status()

            data = response.json()["payload"]

            room_C = (
                data["zon_reaTRooAir_y"] - 273.15
            )

            print(
                f"Hour {data['time']/3600:4.0f} | "
                f"Room={room_C:6.2f} °C | "
                f"Fan={data['fcu_oveFan_u']:.3f} | "
                f"Fan activate="
                f"{data['fcu_oveFan_activate']}"
            )

        # ----------------------------------------------------
        # 4. Override the fan
        #
        # This tells BOPTEST:
        #
        # Ignore the built-in fan command
        # and force normalized fan command = 0.5
        # ----------------------------------------------------

        print()
        print("MANUAL OVERRIDE")
        print("--------------------------------")

        action = {
            "fcu_oveFan_activate": 1,
            "fcu_oveFan_u": 0.5,
        }

        response = requests.post(
            f"{BASE_URL}/advance/{testid}",
            json=action,
        )

        response.raise_for_status()

        data = response.json()["payload"]

        room_C = (
            data["zon_reaTRooAir_y"] - 273.15
        )

        print(
            f"Hour {data['time']/3600:4.0f} | "
            f"Room={room_C:6.2f} °C"
        )

        print(
            f"Fan override command: "
            f"{data['fcu_oveFan_u']}"
        )

        print(
            f"Fan activate: "
            f"{data['fcu_oveFan_activate']}"
        )

        print(
            f"Actual fan electrical power: "
            f"{data['fcu_reaPFan_y']:.2f} W"
        )

        print(
            f"Actual supply flow: "
            f"{data['fcu_reaFloSup_y']:.4f} kg/s"
        )

        # ----------------------------------------------------
        # 5. Release control back to BOPTEST
        # ----------------------------------------------------

        print()
        print("RELEASE OVERRIDE")
        print("--------------------------------")

        response = requests.post(
            f"{BASE_URL}/advance/{testid}",
            json={
                "fcu_oveFan_activate": 0,
            },
        )

        response.raise_for_status()

        data = response.json()["payload"]

        print(
            f"Hour {data['time']/3600:4.0f} | "
            f"Fan activate="
            f"{data['fcu_oveFan_activate']} | "
            f"Fan power="
            f"{data['fcu_reaPFan_y']:.2f} W"
        )

    finally:

        requests.put(
            f"{BASE_URL}/stop/{testid}"
        )

        print()
        print("Test stopped.")


if __name__ == "__main__":
    main()
