import requests, time

BASE = "http://127.0.0.1:8000"

t = time.perf_counter()
r = requests.post(f"{BASE}/testcases/bestest_air/select", timeout=600)
elapsed = time.perf_counter() - t
testid = r.json()["testid"]
print(f"Select: {elapsed:.1f}s testid={testid}")

requests.put(f"{BASE}/stop/{testid}", timeout=30)
