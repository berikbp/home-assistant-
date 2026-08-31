import json
from pathlib import Path

import pandas as pd


DATA_DIR = Path("data/baseline_14d")

TRAJECTORY_FILE = DATA_DIR / "trajectory.csv"
SCHEDULE_FILE = DATA_DIR / "comfort_occupancy_schedule.csv"
KPI_FILE = DATA_DIR / "official_kpis.json"

CONTROL_STEP = 3600
STEP_HOURS = CONTROL_STEP / 3600.0


def main():

    # ========================================================
    # Load existing BOPTEST results
    # ========================================================

    trajectory = pd.read_csv(TRAJECTORY_FILE)
    schedule = pd.read_csv(SCHEDULE_FILE)

    with open(KPI_FILE) as f:
        official_kpis = json.load(f)

    print(f"Trajectory rows: {len(trajectory)}")
    print(f"Schedule points: {len(schedule)}")

    # We expect:
    #
    # 336 simulation intervals
    # 337 forecast boundaries

    assert len(trajectory) == 336
    assert len(schedule) == 337


    # ========================================================
    # Correct time alignment
    #
    # Measurement at time t is the state after simulating:
    #
    #     t - control_step  -->  t
    #
    # Therefore associate it with the schedule at the
    # BEGINNING of that interval.
    # ========================================================

    trajectory["interval_start_time"] = (
        trajectory["time"] - CONTROL_STEP
    )

    schedule_for_merge = schedule[
        [
            "time",
            "lower_setpoint_C",
            "upper_setpoint_C",
            "occupancy",
        ]
    ].copy()

    schedule_for_merge = schedule_for_merge.rename(
        columns={
            "time": "interval_start_time",
            "lower_setpoint_C": "correct_lower_setpoint_C",
            "upper_setpoint_C": "correct_upper_setpoint_C",
            "occupancy": "correct_occupancy",
        }
    )

    df = trajectory.merge(
        schedule_for_merge,
        on="interval_start_time",
        how="left",
        validate="many_to_one",
    )

    if df[
        [
            "correct_lower_setpoint_C",
            "correct_upper_setpoint_C",
            "correct_occupancy",
        ]
    ].isna().any().any():

        raise RuntimeError(
            "Some trajectory rows could not be matched "
            "to the comfort schedule."
        )


    # ========================================================
    # Correct occupancy
    # ========================================================

    df["occupied_corrected"] = (
        df["correct_occupancy"] > 0
    ).astype(int)


    # ========================================================
    # Correct comfort calculation
    # ========================================================

    room = df["room_temperature_C"]

    lower = df["correct_lower_setpoint_C"]
    upper = df["correct_upper_setpoint_C"]

    df["comfortable_corrected"] = (
        (room >= lower)
        & (room <= upper)
    ).astype(int)


    # ========================================================
    # Degree violation
    # ========================================================

    lower_violation = (
        lower - room
    ).clip(lower=0)

    upper_violation = (
        room - upper
    ).clip(lower=0)

    df["sampled_discomfort_corrected_C"] = (
        lower_violation
        + upper_violation
    )


    # ========================================================
    # Comfort Score — 24/7
    # ========================================================

    comfort_24h = (
        100.0
        * df["comfortable_corrected"].mean()
    )


    # ========================================================
    # Occupied-only subset
    # ========================================================

    occupied_df = df[
        df["occupied_corrected"] == 1
    ]

    comfort_occupied = (
        100.0
        * occupied_df["comfortable_corrected"].mean()
    )


    # ========================================================
    # Sampled thermal discomfort
    #
    # This is still NOT the official BOPTEST KPI.
    # It is an hourly approximation.
    # ========================================================

    discomfort_24h = (
        df["sampled_discomfort_corrected_C"].sum()
        * STEP_HOURS
    )

    discomfort_occupied = (
        occupied_df[
            "sampled_discomfort_corrected_C"
        ].sum()
        * STEP_HOURS
    )


    # ========================================================
    # Recalculate occupied sampled HVAC energy
    #
    # NOTE:
    # This remains a CUSTOM metric.
    # Official energy comparison uses BOPTEST ener_tot.
    # ========================================================

    if "sampled_hvac_energy_kWh" in df.columns:

        sampled_energy_total = (
            df["sampled_hvac_energy_kWh"].sum()
        )

        sampled_energy_occupied = (
            df.loc[
                df["occupied_corrected"] == 1,
                "sampled_hvac_energy_kWh",
            ].sum()
        )

    else:

        sampled_energy_total = None
        sampled_energy_occupied = None


    # ========================================================
    # Useful schedule information
    # ========================================================

    occupied_intervals = int(
        df["occupied_corrected"].sum()
    )

    occupied_hours = (
        occupied_intervals
        * STEP_HOURS
    )


    # ========================================================
    # Save corrected trajectory
    # ========================================================

    corrected_file = (
        DATA_DIR / "trajectory_corrected.csv"
    )

    df.to_csv(
        corrected_file,
        index=False,
    )


    # ========================================================
    # Save corrected summary
    # ========================================================

    summary = {
        "method": "BOPTEST_baseline",

        "control_step_seconds":
            CONTROL_STEP,

        "evaluation_intervals":
            len(df),

        "occupied_intervals":
            occupied_intervals,

        "occupied_hours":
            occupied_hours,

        "comfort_score_24h_percent":
            comfort_24h,

        "comfort_score_occupied_percent":
            comfort_occupied,

        "sampled_discomfort_24h_Kh":
            discomfort_24h,

        "sampled_discomfort_occupied_Kh":
            discomfort_occupied,

        "sampled_hvac_energy_total_kWh":
            sampled_energy_total,

        "sampled_hvac_energy_occupied_kWh":
            sampled_energy_occupied,

        "official_tdis_tot":
            official_kpis.get("tdis_tot"),

        "official_idis_tot":
            official_kpis.get("idis_tot"),

        "official_ener_tot":
            official_kpis.get("ener_tot"),

        "official_cost_tot":
            official_kpis.get("cost_tot"),

        "official_emis_tot":
            official_kpis.get("emis_tot"),

        "official_time_rat":
            official_kpis.get("time_rat"),
    }

    summary_file = (
        DATA_DIR / "summary_corrected.json"
    )

    with open(
        summary_file,
        "w",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )


    # ========================================================
    # Print results
    # ========================================================

    print()
    print("========================================")
    print("CORRECTED BASELINE METRICS")
    print("========================================")

    print(
        f"Occupied intervals : "
        f"{occupied_intervals}"
    )

    print(
        f"Occupied hours     : "
        f"{occupied_hours:.1f}"
    )

    print()

    print(
        f"24/7 Comfort Score : "
        f"{comfort_24h:.2f} %"
    )

    print(
        f"Occupied Comfort   : "
        f"{comfort_occupied:.2f} %"
    )

    print()

    print(
        f"Sampled discomfort 24/7 : "
        f"{discomfort_24h:.3f} K·h"
    )

    print(
        f"Sampled occupied discomfort : "
        f"{discomfort_occupied:.3f} K·h"
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
            f"{official_kpis.get(key)}"
        )

    print()
    print(
        f"Corrected trajectory saved to: "
        f"{corrected_file}"
    )

    print(
        f"Corrected summary saved to: "
        f"{summary_file}"
    )


if __name__ == "__main__":
    main()
