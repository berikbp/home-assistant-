import json
import time

import pandas as pd
import requests


BASE_URL = "http://127.0.0.1:8000"


# --------------------------------------------------
# 1. Select bestest_air
# --------------------------------------------------
response = requests.post(
    f"{BASE_URL}/testcases/bestest_air/select"
)
response.raise_for_status()

testid = response.json()["testid"]

print(f"Test ID: {testid}")


# --------------------------------------------------
# 2. Initialize at simulation time 0
# --------------------------------------------------
response = requests.put(
    f"{BASE_URL}/initialize/{testid}",
    json={
        "start_time": 0,
        "warmup_period": 0,
    },
)
response.raise_for_status()

print("Simulation initialized.")


# --------------------------------------------------
# 3. Run 24 one-hour steps
# --------------------------------------------------
rows = []

real_start = time.perf_counter()

for step in range(24):

    response = requests.post(
        f"{BASE_URL}/advance/{testid}",
        json={},
    )
    response.raise_for_status()

    result = response.json()

    if result["status"] != 200:
        raise RuntimeError(result)

    data = result["payload"]

    rows.append(data)

    room_c = data["zon_reaTRooAir_y"] - 273.15
    outdoor_c = data["zon_weaSta_reaWeaTDryBul_y"] - 273.15

    print(
        f"Step {step + 1:02d} | "
        f"hour={data['time'] / 3600:5.1f} | "
        f"room={room_c:6.2f} °C | "
        f"outdoor={outdoor_c:6.2f} °C | "
        f"heat={data['fcu_reaPHea_y']:8.1f} W | "
        f"cool={data['fcu_reaPCoo_y']:8.1f} W | "
        f"fan={data['fcu_reaPFan_y']:8.1f} W"
    )

real_elapsed = time.perf_counter() - real_start


# --------------------------------------------------
# 4. Save trajectory
# --------------------------------------------------
df = pd.DataFrame(rows)

df["room_temperature_C"] = (
    df["zon_reaTRooAir_y"] - 273.15
)

df["outdoor_temperature_C"] = (
    df["zon_weaSta_reaWeaTDryBul_y"] - 273.15
)

df.to_csv(
    "data/baseline_24h.csv",
    index=False,
)


# --------------------------------------------------
# 5. Retrieve official BOPTEST KPIs
# --------------------------------------------------
response = requests.get(
    f"{BASE_URL}/kpi/{testid}"
)
response.raise_for_status()

kpi = response.json()

with open("data/baseline_24h_kpi.json", "w") as f:
    json.dump(kpi, f, indent=2)


# --------------------------------------------------
# 6. Summary
# --------------------------------------------------
print()
print("--------------------------------")
print("24-hour baseline finished")
print("--------------------------------")

print(f"Real runtime: {real_elapsed:.2f} seconds")

print(
    f"Room temperature range: "
    f"{df['room_temperature_C'].min():.2f} °C "
    f"to "
    f"{df['room_temperature_C'].max():.2f} °C"
)

print()
print("Official BOPTEST KPI response:")
print(json.dumps(kpi, indent=2))
