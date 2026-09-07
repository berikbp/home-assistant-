import requests
import time

BASE = "http://127.0.0.1:8000"

# Test select
t = time.perf_counter()
r = requests.post(f"{BASE}/testcases/bestest_air/select", timeout=300)
testid = r.json()["testid"]
print(f"Select: {time.perf_counter()-t:.1f}s testid={testid}")

# Test initialize
t = time.perf_counter()
r = requests.put(f"{BASE}/initialize/{testid}", json={"start_time": 0, "warmup_period": 0}, timeout=300)
print(f"Init: {time.perf_counter()-t:.1f}s status={r.status_code}")

# Test advance
t = time.perf_counter()
r = requests.post(f"{BASE}/advance/{testid}", json={}, timeout=300)
print(f"Advance: {time.perf_counter()-t:.1f}s status={r.status_code}")
data = r.json()["payload"]
room_c = data["zon_reaTRooAir_y"] - 273.15
print(f"Room temp: {room_c:.2f} C")

# Stop
requests.put(f"{BASE}/stop/{testid}", timeout=30)
print("Done - BOPTEST is working!")
