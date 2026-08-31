import matplotlib

# Use a non-interactive backend.
# This prevents Matplotlib from opening GUI windows.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


INPUT_FILE = "data/baseline_24h.csv"


def main():
    # ---------------------------------------------------------
    # Load data
    # ---------------------------------------------------------
    df = pd.read_csv(INPUT_FILE)

    hours = df["time"] / 3600.0

    print(f"Loaded {len(df)} simulation steps.")
    print(
        f"Simulation period: "
        f"{hours.min():.1f} h to {hours.max():.1f} h"
    )

    # ---------------------------------------------------------
    # Plot 1: Indoor and outdoor temperature
    # ---------------------------------------------------------
    fig = plt.figure(figsize=(10, 5))

    plt.plot(
        hours,
        df["room_temperature_C"],
        label="Room temperature",
    )

    plt.plot(
        hours,
        df["outdoor_temperature_C"],
        label="Outdoor temperature",
    )

    plt.xlabel("Simulation time [hour]")
    plt.ylabel("Temperature [°C]")
    plt.title("BOPTEST bestest_air — 24-hour baseline temperatures")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    temperature_output = "data/baseline_24h_temperature.png"

    plt.savefig(
        temperature_output,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {temperature_output}")

    # ---------------------------------------------------------
    # Plot 2: HVAC power
    # ---------------------------------------------------------
    fig = plt.figure(figsize=(10, 5))

    plt.plot(
        hours,
        df["fcu_reaPHea_y"],
        label="Heating power",
    )

    plt.plot(
        hours,
        df["fcu_reaPCoo_y"],
        label="Cooling power",
    )

    plt.plot(
        hours,
        df["fcu_reaPFan_y"],
        label="Fan power",
    )

    plt.xlabel("Simulation time [hour]")
    plt.ylabel("Power [W]")
    plt.title("BOPTEST bestest_air — 24-hour baseline HVAC operation")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    power_output = "data/baseline_24h_power.png"

    plt.savefig(
        power_output,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {power_output}")

    # ---------------------------------------------------------
    # Console summary
    # ---------------------------------------------------------
    print()
    print("----------------------------------------")
    print("Baseline plotting complete")
    print("----------------------------------------")

    print(
        f"Room temperature range: "
        f"{df['room_temperature_C'].min():.2f} °C "
        f"to {df['room_temperature_C'].max():.2f} °C"
    )

    print(
        f"Outdoor temperature range: "
        f"{df['outdoor_temperature_C'].min():.2f} °C "
        f"to {df['outdoor_temperature_C'].max():.2f} °C"
    )

    print(
        f"Maximum heating power: "
        f"{df['fcu_reaPHea_y'].max():.2f} W"
    )

    print(
        f"Maximum cooling power: "
        f"{df['fcu_reaPCoo_y'].max():.2f} W"
    )

    print(
        f"Maximum fan power: "
        f"{df['fcu_reaPFan_y'].max():.2f} W"
    )


if __name__ == "__main__":
    main()