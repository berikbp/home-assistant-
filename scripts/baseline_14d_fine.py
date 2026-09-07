import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests


BASE_URL = "http://127.0.0.1:8000"

TESTCASE = "bestest_air"

START_TIME = 0
WARMUP_PERIOD = 0

CONTROL_STEP = 3600
OUTPUT_INTERVAL = 30

EVALUATION_DAYS = 14
EVALUATION_SECONDS = (
    EVALUATION_DAYS
    * 24
    * 3600
)

N_CONTROL_STEPS = (
    EVALUATION_SECONDS
    // CONTROL_STEP
)

OUTPUT_DIR = Path(
    "data/baseline_14d_fine"
)


# ============================================================
# HTTP helper
# ============================================================

def request_json(
    method,
    endpoint,
    **kwargs,
):

    response = requests.request(
        method,
        f"{BASE_URL}/{endpoint}",
        timeout=180,
        **kwargs,
    )

    response.raise_for_status()

    result = response.json()

    if (
        isinstance(result, dict)
        and result.get("status") is not None
        and result.get("status") != 200
    ):
        raise RuntimeError(
            f"BOPTEST error:\n{result}"
        )

    return result


# ============================================================
# Thermal discomfort
# ============================================================

def comfort_violation(
    temperature_C,
    lower_C,
    upper_C,
):

    lower_violation = np.maximum(
        lower_C - temperature_C,
        0.0,
    )

    upper_violation = np.maximum(
        temperature_C - upper_C,
        0.0,
    )

    return (
        lower_violation
        + upper_violation
    )


# ============================================================
# Main
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 1. Select test case
    # --------------------------------------------------------

    result = request_json(
        "POST",
        f"testcases/{TESTCASE}/select",
    )

    testid = result["testid"]

    print(
        f"Test ID: {testid}"
    )

    try:

        # ----------------------------------------------------
        # 2. Record configuration
        # ----------------------------------------------------

        version = request_json(
            "GET",
            "version",
        )

        scenario = request_json(
            "GET",
            f"scenario/{testid}",
        )

        step_info = request_json(
            "GET",
            f"step/{testid}",
        )

        name_info = request_json(
            "GET",
            f"name/{testid}",
        )

        actual_step = int(
            step_info["payload"]
        )

        if actual_step != CONTROL_STEP:

            raise RuntimeError(
                f"Expected BOPTEST control step "
                f"{CONTROL_STEP} s, "
                f"but server reports "
                f"{actual_step} s."
            )

        configuration = {
            "testcase":
                TESTCASE,

            "start_time_seconds":
                START_TIME,

            "warmup_period_seconds":
                WARMUP_PERIOD,

            "control_step_seconds":
                CONTROL_STEP,

            "stored_output_interval_seconds":
                OUTPUT_INTERVAL,

            "evaluation_days":
                EVALUATION_DAYS,

            "evaluation_seconds":
                EVALUATION_SECONDS,

            "number_of_control_actions":
                N_CONTROL_STEPS,

            "server_version":
                version,

            "scenario":
                scenario,

            "step":
                step_info,

            "name":
                name_info,
        }

        with open(
            OUTPUT_DIR
            / "configuration.json",
            "w",
        ) as f:

            json.dump(
                configuration,
                f,
                indent=2,
            )

        # ----------------------------------------------------
        # 3. Initialize
        # ----------------------------------------------------

        init_result = request_json(
            "PUT",
            f"initialize/{testid}",
            json={
                "start_time":
                    START_TIME,

                "warmup_period":
                    WARMUP_PERIOD,
            },
        )

        print(
            "Simulation initialized."
        )

        # ----------------------------------------------------
        # 4. Retrieve FINE schedule
        #
        # 30-second schedule trajectory.
        #
        # BOPTEST will forward-fill discrete schedule data.
        # ----------------------------------------------------

        forecast_result = request_json(
            "PUT",
            f"forecast/{testid}",
            json={
                "point_names": [
                    "LowerSetp[1]",
                    "UpperSetp[1]",
                    "Occupancy[1]",
                ],

                "horizon":
                    EVALUATION_SECONDS,

                "interval":
                    OUTPUT_INTERVAL,
            },
        )

        schedule = pd.DataFrame(
            forecast_result["payload"]
        )

        schedule = schedule.rename(
            columns={
                "LowerSetp[1]":
                    "lower_setpoint_K",

                "UpperSetp[1]":
                    "upper_setpoint_K",

                "Occupancy[1]":
                    "occupancy",
            }
        )

        schedule[
            "lower_setpoint_C"
        ] = (
            schedule[
                "lower_setpoint_K"
            ]
            - 273.15
        )

        schedule[
            "upper_setpoint_C"
        ] = (
            schedule[
                "upper_setpoint_K"
            ]
            - 273.15
        )

        schedule.to_csv(
            OUTPUT_DIR
            / "schedule_30s.csv",
            index=False,
        )

        print(
            f"Retrieved "
            f"{len(schedule)} "
            f"fine schedule points."
        )

        # ----------------------------------------------------
        # 5. Run 336 controller steps
        #
        # Baseline controller:
        # send empty override dictionary.
        # ----------------------------------------------------

        wall_start = time.perf_counter()

        for step in range(
            N_CONTROL_STEPS
        ):

            result = request_json(
                "POST",
                f"advance/{testid}",
                json={},
            )

            data = result["payload"]

            if (
                step == 0
                or (step + 1) % 24 == 0
                or step + 1
                == N_CONTROL_STEPS
            ):

                room_C = (
                    data[
                        "zon_reaTRooAir_y"
                    ]
                    - 273.15
                )

                print(
                    f"Control step "
                    f"{step + 1:3d}/"
                    f"{N_CONTROL_STEPS} | "
                    f"time="
                    f"{data['time']/3600:6.1f} h | "
                    f"room="
                    f"{room_C:6.2f} °C"
                )

        wall_runtime = (
            time.perf_counter()
            - wall_start
        )

        # ----------------------------------------------------
        # 6. Retrieve complete stored 30-second trajectory
        # ----------------------------------------------------

        result = request_json(
            "PUT",
            f"results/{testid}",
            json={
                "point_names": [
                    "zon_reaTRooAir_y",
                    "zon_weaSta_reaWeaTDryBul_y",

                    "fcu_reaPHea_y",
                    "fcu_reaPCoo_y",
                    "fcu_reaPFan_y",
                    "fcu_reaFloSup_y",

                    "con_oveTSetCoo_y",
                    "con_oveTSetHea_y",
                    "fcu_oveFan_y",
                    "fcu_oveTSup_y",
                ],

                "start_time":
                    START_TIME,

                "final_time":
                    START_TIME
                    + EVALUATION_SECONDS,
            },
        )

        trajectory = pd.DataFrame(
            result["payload"]
        )

        print()
        print(
            f"Stored trajectory points: "
            f"{len(trajectory)}"
        )

        # ----------------------------------------------------
        # 7. Verify stored resolution
        # ----------------------------------------------------

        times = trajectory[
            "time"
        ].to_numpy(
            dtype=float
        )

        dt = np.diff(times)

        print(
            f"Stored time spacing "
            f"(median): "
            f"{np.median(dt):.1f} s"
        )

        if not np.allclose(
            dt,
            OUTPUT_INTERVAL,
        ):

            print(
                "WARNING: stored trajectory "
                "is not uniformly 30 seconds."
            )

        # ----------------------------------------------------
        # 8. Convert useful quantities
        # ----------------------------------------------------

        trajectory[
            "room_temperature_C"
        ] = (
            trajectory[
                "zon_reaTRooAir_y"
            ]
            - 273.15
        )

        trajectory[
            "outdoor_temperature_C"
        ] = (
            trajectory[
                "zon_weaSta_reaWeaTDryBul_y"
            ]
            - 273.15
        )

        # ----------------------------------------------------
        # 9. Merge fine schedule with physical trajectory
        # ----------------------------------------------------

        fine = trajectory.merge(
            schedule[
                [
                    "time",
                    "lower_setpoint_C",
                    "upper_setpoint_C",
                    "occupancy",
                ]
            ],
            on="time",
            how="left",
            validate="one_to_one",
        )

        if fine[
            [
                "lower_setpoint_C",
                "upper_setpoint_C",
                "occupancy",
            ]
        ].isna().any().any():

            raise RuntimeError(
                "Fine trajectory could not "
                "be fully matched to "
                "the BOPTEST schedule."
            )

        # ----------------------------------------------------
        # 10. Fine-resolution thermal comfort
        # ----------------------------------------------------

        fine[
            "discomfort_K"
        ] = comfort_violation(
            fine[
                "room_temperature_C"
            ].to_numpy(),

            fine[
                "lower_setpoint_C"
            ].to_numpy(),

            fine[
                "upper_setpoint_C"
            ].to_numpy(),
        )

        fine[
            "comfortable"
        ] = (
            fine[
                "discomfort_K"
            ]
            <= 1e-12
        ).astype(int)

        fine[
            "occupied"
        ] = (
            fine[
                "occupancy"
            ]
            > 0
        ).astype(int)

        # ----------------------------------------------------
        # 11. Time-weighted Comfort Score
        #
        # Use interval start points:
        #
        # [t_i, t_{i+1})
        #
        # This avoids giving the final endpoint an extra
        # 30-second weight.
        # ----------------------------------------------------

        interval_df = fine.iloc[
            :-1
        ].copy()

        interval_df[
            "dt_seconds"
        ] = np.diff(
            fine["time"].to_numpy()
        )

        total_seconds = (
            interval_df[
                "dt_seconds"
            ].sum()
        )

        comfortable_seconds = (
            interval_df.loc[
                interval_df[
                    "comfortable"
                ] == 1,
                "dt_seconds",
            ].sum()
        )

        comfort_score_24h = (
            100.0
            * comfortable_seconds
            / total_seconds
        )

        occupied_intervals = (
            interval_df[
                "occupied"
            ]
            == 1
        )

        occupied_seconds = (
            interval_df.loc[
                occupied_intervals,
                "dt_seconds",
            ].sum()
        )

        occupied_comfort_seconds = (
            interval_df.loc[
                occupied_intervals
                & (
                    interval_df[
                        "comfortable"
                    ]
                    == 1
                ),
                "dt_seconds",
            ].sum()
        )

        if occupied_seconds > 0:

            comfort_score_occupied = (
                100.0
                * occupied_comfort_seconds
                / occupied_seconds
            )

        else:

            comfort_score_occupied = None

        occupied_hours = (
            occupied_seconds
            / 3600.0
        )

        # ----------------------------------------------------
        # 12. Fine custom discomfort
        #
        # This uses the same basic trapezoidal numerical
        # form as BOPTEST's official tdis calculation.
        #
        # It remains labelled CUSTOM because official
        # tdis_tot comes directly from BOPTEST.
        # ----------------------------------------------------

        fine_discomfort_Kh = float(
            np.trapezoid(
                fine[
                    "discomfort_K"
                ].to_numpy(),

                fine[
                    "time"
                ].to_numpy(),
            )
            / 3600.0
        )

        # Occupied-only custom discomfort
        #
        # Multiply the violation by occupancy indicator
        # before integration.
        occupied_discomfort_signal = (
            fine[
                "discomfort_K"
            ].to_numpy()
            * fine[
                "occupied"
            ].to_numpy()
        )

        fine_occupied_discomfort_Kh = float(
            np.trapezoid(
                occupied_discomfort_signal,
                fine[
                    "time"
                ].to_numpy(),
            )
            / 3600.0
        )

        # ----------------------------------------------------
        # 13. Fine custom physical component energies
        #
        # Diagnostic only.
        # Do not replace BOPTEST ener_tot.
        # ----------------------------------------------------

        time_seconds = fine[
            "time"
        ].to_numpy()

        heating_kWh = float(
            np.trapezoid(
                fine[
                    "fcu_reaPHea_y"
                ].to_numpy(),
                time_seconds,
            )
            / 3_600_000.0
        )

        cooling_kWh = float(
            np.trapezoid(
                fine[
                    "fcu_reaPCoo_y"
                ].to_numpy(),
                time_seconds,
            )
            / 3_600_000.0
        )

        fan_kWh = float(
            np.trapezoid(
                fine[
                    "fcu_reaPFan_y"
                ].to_numpy(),
                time_seconds,
            )
            / 3_600_000.0
        )

        custom_total_energy_kWh = (
            heating_kWh
            + cooling_kWh
            + fan_kWh
        )

        # ----------------------------------------------------
        # 14. Official BOPTEST KPIs
        # ----------------------------------------------------

        kpi_result = request_json(
            "GET",
            f"kpi/{testid}",
        )

        official_kpis = (
            kpi_result["payload"]
        )

        # ----------------------------------------------------
        # 15. Save everything
        # ----------------------------------------------------

        fine.to_csv(
            OUTPUT_DIR
            / "trajectory_30s.csv",
            index=False,
        )

        with open(
            OUTPUT_DIR
            / "official_kpis.json",
            "w",
        ) as f:

            json.dump(
                official_kpis,
                f,
                indent=2,
            )

        summary = {

            "method":
                "BOPTEST baseline (PI)",

            "evaluation_days":
                EVALUATION_DAYS,

            "control_step_seconds":
                CONTROL_STEP,

            "stored_output_interval_seconds":
                OUTPUT_INTERVAL,

            "number_control_actions":
                N_CONTROL_STEPS,

            "number_fine_points":
                len(fine),

            "occupied_hours_from_boptest_schedule":
                occupied_hours,

            "comfort_score_24h_percent":
                comfort_score_24h,

            "comfort_score_occupied_percent":
                comfort_score_occupied,

            "fine_custom_discomfort_24h_Kh":
                fine_discomfort_Kh,

            "fine_custom_occupied_discomfort_Kh":
                fine_occupied_discomfort_Kh,

            "fine_heating_component_kWh":
                heating_kWh,

            "fine_cooling_component_kWh":
                cooling_kWh,

            "fine_fan_component_kWh":
                fan_kWh,

            "fine_custom_total_component_energy_kWh":
                custom_total_energy_kWh,

            "official_tdis_tot":
                official_kpis.get(
                    "tdis_tot"
                ),

            "official_idis_tot":
                official_kpis.get(
                    "idis_tot"
                ),

            "official_ener_tot":
                official_kpis.get(
                    "ener_tot"
                ),

            "official_cost_tot":
                official_kpis.get(
                    "cost_tot"
                ),

            "official_emis_tot":
                official_kpis.get(
                    "emis_tot"
                ),

            "official_time_rat":
                official_kpis.get(
                    "time_rat"
                ),

            "wall_runtime_seconds":
                wall_runtime,
        }

        with open(
            OUTPUT_DIR
            / "summary.json",
            "w",
        ) as f:

            json.dump(
                summary,
                f,
                indent=2,
            )

        # ----------------------------------------------------
        # 16. Print final results
        # ----------------------------------------------------

        print()
        print(
            "========================================"
        )

        print(
            "14-DAY FINE BASELINE COMPLETE"
        )

        print(
            "========================================"
        )

        print(
            f"Wall runtime: "
            f"{wall_runtime:.2f} s"
        )

        print()
        print(
            "FINE CUSTOM METRICS"
        )

        print(
            "----------------------------------------"
        )

        print(
            f"Occupied hours from schedule: "
            f"{occupied_hours:.3f} h"
        )

        print(
            f"24/7 Comfort Score: "
            f"{comfort_score_24h:.2f} %"
        )

        print(
            f"Occupied Comfort Score: "
            f"{comfort_score_occupied:.2f} %"
        )

        print(
            f"Fine custom discomfort: "
            f"{fine_discomfort_Kh:.6f} K·h"
        )

        print(
            f"Fine occupied discomfort: "
            f"{fine_occupied_discomfort_Kh:.6f} K·h"
        )

        print()
        print(
            "OFFICIAL BOPTEST KPIs"
        )

        print(
            "----------------------------------------"
        )

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
                f"{official_kpis.get(key)}"
            )

        print()
        print(
            f"Files saved in: "
            f"{OUTPUT_DIR}"
        )

    finally:

        try:

            requests.put(
                f"{BASE_URL}/stop/{testid}",
                timeout=30,
            )

            print(
                f"Stopped test instance: "
                f"{testid}"
            )

        except Exception as exc:

            print(
                f"Warning: unable to stop "
                f"test instance: {exc}"
            )


if __name__ == "__main__":
    main()
