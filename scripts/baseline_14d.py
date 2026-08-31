import json
import time
from pathlib import Path

import pandas as pd
import requests


# ============================================================
# Experiment configuration
# ============================================================

BASE_URL = "http://127.0.0.1:8000"

TESTCASE = "bestest_air"

START_TIME = 0
WARMUP_PERIOD = 0

CONTROL_STEP = 3600       # seconds = 1 hour
EVALUATION_DAYS = 14
N_STEPS = EVALUATION_DAYS * 24

OUTPUT_DIR = Path("data/baseline_14d")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_json(method, url, **kwargs):
    response = requests.request(
        method,
        url,
        timeout=120,
        **kwargs,
    )

    response.raise_for_status()

    result = response.json()

    if isinstance(result, dict):
        status = result.get("status")

        if status is not None and status != 200:
            raise RuntimeError(result)

    return result


def main():

    testid = None

    try:

        # ====================================================
        # 1. Select bestest_air
        # ====================================================

        result = get_json(
            "POST",
            f"{BASE_URL}/testcases/{TESTCASE}/select",
        )

        testid = result["testid"]

        print(f"Test ID: {testid}")


        # ====================================================
        # 2. Record BOPTEST configuration
        # ====================================================

        version = get_json(
            "GET",
            f"{BASE_URL}/version",
        )

        scenario = get_json(
            "GET",
            f"{BASE_URL}/scenario/{testid}",
        )

        step_info = get_json(
            "GET",
            f"{BASE_URL}/step/{testid}",
        )

        testcase_name = get_json(
            "GET",
            f"{BASE_URL}/name/{testid}",
        )

        config = {
            "testcase": TESTCASE,
            "testid": testid,

            "start_time_s": START_TIME,
            "warmup_period_s": WARMUP_PERIOD,

            "control_step_s": CONTROL_STEP,

            "evaluation_days": EVALUATION_DAYS,
            "evaluation_steps": N_STEPS,

            "version_response": version,
            "scenario_response": scenario,
            "step_response": step_info,
            "name_response": testcase_name,
        }

        with open(
            OUTPUT_DIR / "configuration.json",
            "w",
        ) as f:
            json.dump(
                config,
                f,
                indent=2,
            )


        print()
        print("Experiment configuration")
        print("------------------------")
        print(f"Test case       : {TESTCASE}")
        print(f"Start time      : {START_TIME} s")
        print(f"Warm-up         : {WARMUP_PERIOD} s")
        print(f"Control step    : {CONTROL_STEP} s")
        print(f"Evaluation      : {EVALUATION_DAYS} days")
        print(f"Number of steps : {N_STEPS}")

        print()


        # ====================================================
        # 3. Initialize simulation
        # ====================================================

        get_json(
            "PUT",
            f"{BASE_URL}/initialize/{testid}",
            json={
                "start_time": START_TIME,
                "warmup_period": WARMUP_PERIOD,
            },
        )

        print("Simulation initialized.")


        # ====================================================
        # 4. Retrieve complete 14-day comfort/occupancy schedule
        # ====================================================

        horizon = EVALUATION_DAYS * 24 * 3600

        forecast_result = get_json(
            "PUT",
            f"{BASE_URL}/forecast/{testid}",
            json={
                "point_names": [
                    "LowerSetp[1]",
                    "UpperSetp[1]",
                    "Occupancy[1]",
                ],
                "horizon": horizon,
                "interval": CONTROL_STEP,
            },
        )

        forecast = forecast_result["payload"]

        forecast_df = pd.DataFrame(
            {
                "time": forecast["time"],
                "lower_setpoint_K": forecast["LowerSetp[1]"],
                "upper_setpoint_K": forecast["UpperSetp[1]"],
                "occupancy": forecast["Occupancy[1]"],
            }
        )

        forecast_df["lower_setpoint_C"] = (
            forecast_df["lower_setpoint_K"] - 273.15
        )

        forecast_df["upper_setpoint_C"] = (
            forecast_df["upper_setpoint_K"] - 273.15
        )

        forecast_df.to_csv(
            OUTPUT_DIR / "comfort_occupancy_schedule.csv",
            index=False,
        )

        # Make lookup by simulation time.
        schedule_lookup = (
            forecast_df
            .set_index("time")
            .to_dict("index")
        )

        print(
            f"Retrieved {len(forecast_df)} "
            "comfort/occupancy schedule points."
        )


        # ====================================================
        # 5. Run 14-day baseline
        # ====================================================

        rows = []

        real_start = time.perf_counter()

        for step_number in range(1, N_STEPS + 1):

            result = get_json(
                "POST",
                f"{BASE_URL}/advance/{testid}",
                json={},
            )

            data = result["payload"]

            simulation_time = data["time"]

            schedule = schedule_lookup[simulation_time]

            room_C = (
                data["zon_reaTRooAir_y"]
                - 273.15
            )

            outdoor_C = (
                data["zon_weaSta_reaWeaTDryBul_y"]
                - 273.15
            )

            lower_C = schedule["lower_setpoint_C"]
            upper_C = schedule["upper_setpoint_C"]

            occupancy = schedule["occupancy"]

            # -----------------------------------------------
            # Comfort calculation
            # -----------------------------------------------

            lower_violation = max(
                lower_C - room_C,
                0.0,
            )

            upper_violation = max(
                room_C - upper_C,
                0.0,
            )

            discomfort_C = (
                lower_violation
                + upper_violation
            )

            comfortable = (
                lower_C
                <= room_C
                <= upper_C
            )

            occupied = occupancy > 0

            # -----------------------------------------------
            # Add useful columns
            # -----------------------------------------------

            data["room_temperature_C"] = room_C
            data["outdoor_temperature_C"] = outdoor_C

            data["lower_setpoint_C"] = lower_C
            data["upper_setpoint_C"] = upper_C

            data["occupancy"] = occupancy

            data["comfortable"] = int(comfortable)
            data["occupied"] = int(occupied)

            data["sampled_discomfort_C"] = discomfort_C

            rows.append(data)

            # Print once per simulated day
            if (
                step_number == 1
                or step_number % 24 == 0
            ):

                print(
                    f"Step {step_number:3d}/{N_STEPS} | "
                    f"day={simulation_time / 86400:5.1f} | "
                    f"room={room_C:6.2f} °C | "
                    f"outside={outdoor_C:6.2f} °C | "
                    f"occupancy={occupancy}"
                )


        real_elapsed = (
            time.perf_counter()
            - real_start
        )


        # ====================================================
        # 6. Convert trajectory to DataFrame
        # ====================================================

        df = pd.DataFrame(rows)

        df.to_csv(
            OUTPUT_DIR / "trajectory.csv",
            index=False,
        )


        # ====================================================
        # 7. Custom Comfort Score
        # ====================================================

        comfort_score_24h = (
            100.0
            * df["comfortable"].mean()
        )

        occupied_df = df[
            df["occupied"] == 1
        ]

        if len(occupied_df) > 0:

            comfort_score_occupied = (
                100.0
                * occupied_df["comfortable"].mean()
            )

        else:
            comfort_score_occupied = None


        # ====================================================
        # 8. Sampled discomfort
        #
        # IMPORTANT:
        # This is NOT official BOPTEST tdis_tot.
        # Official tdis_tot is retrieved later.
        # ====================================================

        step_hours = (
            CONTROL_STEP / 3600.0
        )

        sampled_discomfort_24h_Kh = (
            df["sampled_discomfort_C"].sum()
            * step_hours
        )

        sampled_discomfort_occupied_Kh = (
            occupied_df["sampled_discomfort_C"].sum()
            * step_hours
        )


        # ====================================================
        # 9. Additional sampled HVAC energy
        #
        # These are derived from the available output signals.
        # They are NOT replacements for official ener_tot.
        # ====================================================

        df["heating_thermal_energy_kWh"] = (
            df["fcu_reaPHea_y"]
            * step_hours
            / 1000.0
        )

        df["cooling_electric_energy_kWh"] = (
            df["fcu_reaPCoo_y"]
            * step_hours
            / 1000.0
        )

        df["fan_electric_energy_kWh"] = (
            df["fcu_reaPFan_y"]
            * step_hours
            / 1000.0
        )

        df["sampled_hvac_energy_kWh"] = (
            df["heating_thermal_energy_kWh"]
            + df["cooling_electric_energy_kWh"]
            + df["fan_electric_energy_kWh"]
        )

        # Save again after energy columns added.
        df.to_csv(
            OUTPUT_DIR / "trajectory.csv",
            index=False,
        )


        total_sampled_energy_kWh = (
            df["sampled_hvac_energy_kWh"].sum()
        )

        occupied_sampled_energy_kWh = (
            df.loc[
                df["occupied"] == 1,
                "sampled_hvac_energy_kWh",
            ].sum()
        )


        # ====================================================
        # 10. Official BOPTEST KPIs
        # ====================================================

        kpi_result = get_json(
            "GET",
            f"{BASE_URL}/kpi/{testid}",
        )

        kpis = kpi_result["payload"]

        with open(
            OUTPUT_DIR / "official_kpis.json",
            "w",
        ) as f:

            json.dump(
                kpis,
                f,
                indent=2,
            )


        # ====================================================
        # 11. Save custom summary
        # ====================================================

        summary = {

            "method": "BOPTEST_baseline",

            "comfort_score_24h_percent":
                comfort_score_24h,

            "comfort_score_occupied_percent":
                comfort_score_occupied,

            "sampled_discomfort_24h_Kh":
                sampled_discomfort_24h_Kh,

            "sampled_discomfort_occupied_Kh":
                sampled_discomfort_occupied_Kh,

            "sampled_hvac_energy_total_kWh":
                total_sampled_energy_kWh,

            "sampled_hvac_energy_occupied_kWh":
                occupied_sampled_energy_kWh,

            "official_tdis_tot":
                kpis.get("tdis_tot"),

            "official_idis_tot":
                kpis.get("idis_tot"),

            "official_ener_tot":
                kpis.get("ener_tot"),

            "official_cost_tot":
                kpis.get("cost_tot"),

            "official_emis_tot":
                kpis.get("emis_tot"),

            "official_time_rat":
                kpis.get("time_rat"),

            "real_runtime_seconds":
                real_elapsed,
        }

        with open(
            OUTPUT_DIR / "summary.json",
            "w",
        ) as f:

            json.dump(
                summary,
                f,
                indent=2,
            )


        # ====================================================
        # 12. Print final results
        # ====================================================

        print()
        print("========================================")
        print("14-DAY BOPTEST BASELINE COMPLETE")
        print("========================================")

        print(
            f"Real runtime: "
            f"{real_elapsed:.2f} s"
        )

        print()
        print("CUSTOM METRICS")
        print("----------------------------------------")

        print(
            f"24/7 Comfort Score: "
            f"{comfort_score_24h:.2f} %"
        )

        if comfort_score_occupied is not None:

            print(
                f"Occupied Comfort Score: "
                f"{comfort_score_occupied:.2f} %"
            )

        print(
            f"Sampled 24/7 discomfort: "
            f"{sampled_discomfort_24h_Kh:.3f} K·h"
        )

        print(
            f"Sampled occupied discomfort: "
            f"{sampled_discomfort_occupied_Kh:.3f} K·h"
        )

        print(
            f"Sampled total HVAC energy: "
            f"{total_sampled_energy_kWh:.3f} kWh"
        )

        print(
            f"Sampled occupied HVAC energy: "
            f"{occupied_sampled_energy_kWh:.3f} kWh"
        )


        print()
        print("OFFICIAL BOPTEST KPIs")
        print("----------------------------------------")

        for key in [
            "tdis_tot",
            "idis_tot",
            "ener_tot",
            "cost_tot",
            "emis_tot",
            "time_rat",
        ]:

            print(
                f"{key:10s}: "
                f"{kpis.get(key)}"
            )


        print()
        print(
            f"Files saved in: "
            f"{OUTPUT_DIR}"
        )


    finally:

        # ====================================================
        # 13. Always free the BOPTEST worker
        # ====================================================

        if testid is not None:

            try:

                requests.put(
                    f"{BASE_URL}/stop/{testid}",
                    timeout=30,
                )

                print(
                    f"\nStopped test instance: {testid}"
                )

            except Exception as exc:

                print(
                    "\nWarning: could not stop "
                    f"test instance: {exc}"
                )


if __name__ == "__main__":
    main()

