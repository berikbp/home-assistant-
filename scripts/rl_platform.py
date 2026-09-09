#!/usr/bin/env python3
"""HVAC RL Control Platform — self-contained web dashboard with persistent agent storage."""

import json, sys, math, threading, time, os, shutil, io, csv
from datetime import datetime
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.parse import urlparse, unquote

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC, PPO, DDPG
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor


MODELS_DIR = Path(__file__).parent.parent / "models"
ELECTRICITY_PRICE = 0.12  # $/kWh


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


ALGO_MAP = {"SAC": SAC, "PPO": PPO, "DDPG": DDPG}


def save_agent(name, algo, model, metrics, train_cfg):
    d = MODELS_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    model.save(str(d / "model"))
    meta = {
        "name": name, "algo": algo,
        "saved_at": datetime.now().isoformat(),
        "train_cfg": train_cfg,
        "metrics": metrics,
    }
    (d / "meta.json").write_text(json.dumps(meta, indent=2, cls=NumpyEncoder))
    return meta


def list_agents():
    if not MODELS_DIR.exists():
        return []
    agents = []
    for d in sorted(MODELS_DIR.iterdir()):
        mf = d / "meta.json"
        if mf.exists():
            agents.append(json.loads(mf.read_text()))
    return agents


def load_agent_meta(name):
    mf = MODELS_DIR / name / "meta.json"
    if not mf.exists():
        return None
    return json.loads(mf.read_text())


def load_agent_model(name):
    meta = load_agent_meta(name)
    if meta is None:
        return None, None
    algo = meta["algo"]
    cls = ALGO_MAP.get(algo)
    if cls is None:
        return None, None
    model = cls.load(str(MODELS_DIR / name / "model"))
    return model, meta


def delete_agent(name):
    d = MODELS_DIR / name
    if d.exists():
        shutil.rmtree(d)
        return True
    return False


class ThermalModel:
    def __init__(self, C=2.0, R=5.0, alpha=0.15, A=48.0, K_u=0.3):
        self.C, self.R, self.alpha, self.A, self.K_u = C, R, alpha, A, K_u

    def step(self, Tin, Tout, G, fan, Tsup, dt=5.0/60.0):
        Qw = (Tout - Tin) / self.R
        Qs = self.alpha * G * self.A / 1000.0
        Qh = fan * self.K_u * (Tsup - Tin)
        Tin2 = Tin + (Qw + Qs + Qh) / self.C * dt
        Ph = max(Qh, 0) * 1000
        Pc = max(-Qh, 0) * 1000
        Pf = fan * 50
        return Tin2, Ph, Pc, Pf


class SimpleHVACEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, weather, energy_weight=0.6, episode_steps=None,
                 timestep_min=5, comfort_deadband=None, setpoint_lower=None, setpoint_upper=None,
                 occ_hours=None, occ_setpoint_lower=None, occ_setpoint_upper=None,
                 unocc_setpoint_lower=None, unocc_setpoint_upper=None,
                 building_C=2.0, building_R=5.0, building_alpha=0.15, building_A=48.0, building_K_u=0.3):
        super().__init__()
        self.timestep_min = timestep_min
        self.dt = timestep_min / 60.0
        self.thermal = ThermalModel(building_C, building_R, building_alpha, building_A, building_K_u)
        self.weather = weather
        self.ew = energy_weight
        steps_per_hour = 60 / timestep_min
        self.ep = episode_steps if episode_steps else int(14 * 24 * steps_per_hour)
        total_steps = len(weather)
        self.dref = 6.8247 / total_steps
        self.eref = 4.2263 / total_steps
        self.fa = building_A
        self.action_space = spaces.Box(-1, 1, (2,), np.float32)
        self.observation_space = spaces.Box(
            np.array([-20, -50, 0, 0, 0, 0, -1, -1], np.float32),
            np.array([60, 60, 1500, 50, 50, 20, 1, 1], np.float32))
        self.step_n = 0
        self.Tin = 20.0
        self.occ_hours = occ_hours if occ_hours is not None else list(range(8, 20))
        self.occ_lo = occ_setpoint_lower if occ_setpoint_lower is not None else 21
        self.occ_hi = occ_setpoint_upper if occ_setpoint_upper is not None else 24
        self.unocc_lo = unocc_setpoint_lower if unocc_setpoint_lower is not None else 15
        self.unocc_hi = unocc_setpoint_upper if unocc_setpoint_upper is not None else 30

    def _w(self, s):
        r = self.weather[s % len(self.weather)]
        return r[0], r[1]

    def _sch(self, s):
        h = s % 24
        if h in self.occ_hours:
            return (self.occ_lo, self.occ_hi, 1)
        return (self.unocc_lo, self.unocc_hi, 0)

    def _obs(self):
        To, G = self._w(self.step_n)
        lo, hi, oc = self._sch(self.step_n)
        a = 2 * math.pi * (self.step_n % 24) / 24
        return np.array([self.Tin, To, G, lo, hi, oc, math.sin(a), math.cos(a)], np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_n = 0
        self.Tin = 20.0 + np.random.uniform(-2, 2)
        return self._obs(), {}

    def step(self, action):
        a = np.clip(action, -1, 1)
        fan = (a[0] + 1) / 2
        Tsup = 12 + (a[1] + 1) / 2 * 28
        To, G = self._w(self.step_n)
        self.Tin, Ph, Pc, Pf = self.thermal.step(self.Tin, To, G, fan, Tsup, self.dt)
        lo, hi, _ = self._sch(self.step_n)
        dis = max(lo - self.Tin, 0) + max(self.Tin - hi, 0)
        dn = dis / self.dref
        en = (Ph + Pc + Pf) / 1000 / self.fa / self.eref
        reward = -(dn + self.ew * en)
        self.step_n += 1
        done = self.step_n >= self.ep
        info = dict(zone_temperature_c=self.Tin, lower_setpoint_c=lo, upper_setpoint_c=hi,
                    discomfort_proxy=dis, power_heat_w=Ph, power_cool_w=Pc, power_fan_w=Pf,
                    energy_proxy_kwh_m2=(Ph+Pc+Pf)/1000/self.fa,
                    action_fan=float(fan), action_tsup=float(Tsup),
                    outdoor_temp=float(To), solar_irradiance=float(G),
                    is_occupied=int(_ == 1))
        return self._obs(), reward, False, done, info


class PI:
    def __init__(self):
        self.Kp, self.Ki, self.integral = 2.0, 0.5, 0.0

    def reset(self):
        self.integral = 0.0

    def act(self, Tin, lo, hi):
        err = (lo + hi) / 2 - Tin
        self.integral = np.clip(self.integral + err, -10, 10)
        out = self.Kp * err + self.Ki * self.integral
        fan = np.clip(abs(out) / 5, 0, 1)
        Tsup = np.clip(20 + out * 2, 12, 40)
        return np.array([fan * 2 - 1, (Tsup - 12) / 28 * 2 - 1], np.float32)


class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.reset()
    def reset(self):
        with self.lock:
            self.training = False
            self.algo = None
            self.total = 0
            self.current = 0
            self.rewards = []
            self.comforts = []
            self.energies = []
            self.model = None
            self.status = "idle"
            self.error = None
            self.last_saved = None
    def update(self, **kw):
        with self.lock:
            for k, v in kw.items():
                setattr(self, k, v)
    def get(self, k):
        with self.lock:
            return getattr(self, k)


class CB(BaseCallback):
    def __init__(self, st, verbose=0):
        super().__init__(verbose)
        self.st = st
        self._r, self._d, self._e = 0, 0, 0

    def _on_step(self):
        self._r += self.locals["rewards"][0]
        info = self.locals["infos"][0]
        self._d += info.get("discomfort_proxy", 0)
        self._e += info.get("energy_proxy_kwh_m2", 0)
        if self.locals["dones"][0]:
            with self.st.lock:
                self.st.rewards.append(self._r)
                self.st.comforts.append(self._d)
                self.st.energies.append(self._e)
                self.st.current = self.num_timesteps
            self._r = self._d = self._e = 0
        return True


def load_weather(timestep_min=5):
    csv_path = Path(__file__).parent.parent / "data" / "baseline_14d" / "trajectory.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        Tout = df["zon_weaSta_reaWeaTDryBul_y"].values - 273.15
        G = df["zon_weaSta_reaWeaHGloHor_y"].values
        n_hours = len(Tout)
        steps_per_hour = 60 / timestep_min
        n_steps = int(n_hours * steps_per_hour)
        x_hour = np.arange(n_hours)
        x_new = np.linspace(0, n_hours - 1, n_steps)
        Tout_new = np.interp(x_new, x_hour, Tout)
        G_new = np.interp(x_new, x_hour, G)
        G_new = np.maximum(0, G_new)
        return np.column_stack([Tout_new, G_new])
    steps_per_hour = 60 / timestep_min
    n_steps = int(14 * 24 * steps_per_hour)
    h = np.arange(n_steps)
    Tout = 15 + 5 * np.sin(2*np.pi*h/(24*steps_per_hour) - np.pi/2) + 3*np.sin(2*np.pi*h/n_steps)
    G = np.maximum(0, 600*np.sin(2*np.pi*h/(24*steps_per_hour) - np.pi/3))
    return np.column_stack([Tout, G])


def _env_kwargs(cfg):
    kw = {}
    if cfg.get("occ_hours") is not None:
        kw["occ_hours"] = cfg["occ_hours"]
    if cfg.get("occ_setpoint_lower") is not None:
        kw["occ_setpoint_lower"] = float(cfg["occ_setpoint_lower"])
    if cfg.get("occ_setpoint_upper") is not None:
        kw["occ_setpoint_upper"] = float(cfg["occ_setpoint_upper"])
    if cfg.get("unocc_setpoint_lower") is not None:
        kw["unocc_setpoint_lower"] = float(cfg["unocc_setpoint_lower"])
    if cfg.get("unocc_setpoint_upper") is not None:
        kw["unocc_setpoint_upper"] = float(cfg["unocc_setpoint_upper"])
    if cfg.get("building_C") is not None:
        kw["building_C"] = float(cfg["building_C"])
    if cfg.get("building_R") is not None:
        kw["building_R"] = float(cfg["building_R"])
    if cfg.get("building_alpha") is not None:
        kw["building_alpha"] = float(cfg["building_alpha"])
    if cfg.get("building_A") is not None:
        kw["building_A"] = float(cfg["building_A"])
    if cfg.get("building_K_u") is not None:
        kw["building_K_u"] = float(cfg["building_K_u"])
    return kw


def run_episode(model, weather, ep_steps=None, ew=0.6, env_cfg=None, timestep_min=5):
    kw = _env_kwargs(env_cfg or {})
    env = SimpleHVACEnv(weather, ew, ep_steps, timestep_min=timestep_min, **kw)
    obs, _ = env.reset()
    traj = []
    for _ in range(env.ep):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, _, done, info = env.step(action)
        info["reward"] = float(r)
        info["action_fan"] = float((action[0] + 1) / 2)
        info["action_tsup"] = float(12 + (action[1] + 1) / 2 * 28)
        traj.append(info)
        if done:
            break
    env.close()
    return traj


def run_baseline(weather, ep_steps=None, ew=0.6, env_cfg=None, timestep_min=5):
    kw = _env_kwargs(env_cfg or {})
    env = SimpleHVACEnv(weather, ew, ep_steps, timestep_min=timestep_min, **kw)
    obs, _ = env.reset()
    pi = PI()
    pi.reset()
    traj = []
    lo0, hi0, _ = env._sch(0)
    act = pi.act(env.Tin, lo0, hi0)
    for _ in range(env.ep):
        obs, r, _, done, info = env.step(act)
        lo, hi, _ = env._sch(env.step_n - 1)
        tin = info["zone_temperature_c"]
        act = pi.act(tin, lo, hi)
        info["reward"] = float(r)
        info["action_fan"] = float((act[0] + 1) / 2)
        info["action_tsup"] = float(12 + (act[1] + 1) / 2 * 28)
        traj.append(info)
        if done:
            break
    env.close()
    return traj


def sim_metrics(traj):
    room = np.array([t["zone_temperature_c"] for t in traj])
    lo = np.array([t["lower_setpoint_c"] for t in traj])
    hi = np.array([t["upper_setpoint_c"] for t in traj])
    comf = ((room >= lo) & (room <= hi)).astype(float)
    total_energy_kwh = float(sum(t["energy_proxy_kwh_m2"] for t in traj))
    cost = total_energy_kwh * ELECTRICITY_PRICE
    occupied_mask = np.array([t.get("is_occupied", 1) for t in traj])
    occ_comf = comf[occupied_mask == 1] if occupied_mask.sum() > 0 else comf
    return dict(
        comfort_score=round(float(100 * comf.mean()), 4),
        occupied_comfort_score=round(float(100 * occ_comf.mean()), 4) if len(occ_comf) > 0 else 0,
        total_energy=round(total_energy_kwh, 6),
        total_discomfort=round(float(sum(t["discomfort_proxy"] for t in traj)), 4),
        total_reward=round(float(sum(t["reward"] for t in traj)), 4),
        cost_usd=round(cost, 4),
        energy_cost_per_day=round(cost / max(1, len(traj) / 24), 4),
    )


def traj_to_csv(traj):
    if not traj:
        return ""
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=traj[0].keys())
    w.writeheader()
    w.writerows(traj)
    return buf.getvalue()


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class Handler(BaseHTTPRequestHandler):
    st = None
    weather = None
    html_path = None
    baseline = None
    sweep_state = None

    def log_message(self, *a):
        pass

    def _json(self, data, code=200):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, cls=NumpyEncoder).encode())

    def _html(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(open(self.html_path, "rb").read())

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        p = urlparse(self.path).path
        if p in ("/", "/index.html"):
            return self._html()
        if p == "/api/status":
            s = self.st
            return self._json(dict(
                training=s.get("training"), algo=s.get("algo"),
                total=s.get("total"), current=s.get("current"),
                n_episodes=len(s.get("rewards")),
                status=s.get("status"), error=s.get("error"),
                last_saved=s.get("last_saved")))
        if p == "/api/results":
            rw = self.st.get("rewards")
            if not rw:
                return self._json(dict(has_results=False))
            return self._json(dict(
                has_results=True, algo=self.st.get("algo"),
                episode_rewards=rw, episode_comforts=self.st.get("comforts"),
                episode_energies=self.st.get("energies")))
        if p == "/api/baseline":
            return self._json(self.baseline)
        if p == "/api/agents":
            return self._json(list_agents())
        if p == "/api/sweep-status":
            ss = Handler.sweep_state
            if ss is None:
                return self._json(dict(running=False))
            return self._json(ss)
        if p.startswith("/api/export/"):
            kind = p.split("/api/export/")[1]
            return self._export(kind)
        self.send_error(404)

    def do_POST(self):
        p = urlparse(self.path).path
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        if p == "/api/train":
            return self._train(body)
        if p == "/api/simulate":
            return self._simulate(body)
        if p == "/api/simulate-saved":
            return self._simulate_saved(body)
        if p.startswith("/api/agents/") and p.endswith("/simulate"):
            name = unquote(p.split("/")[3])
            return self._simulate_saved({"name": name, **body})
        if p.startswith("/api/agents/") and p.endswith("/delete"):
            name = unquote(p.split("/")[3])
            delete_agent(name)
            return self._json({"ok": True})
        if p == "/api/agents/compare":
            return self._compare_agents(body)
        if p == "/api/sweep":
            return self._sweep(body)
        self.send_error(404)

    def _export(self, kind):
        rl = self.st.get("last_rl_traj")
        bl = self.st.get("last_bl_traj")
        fmt = "csv"
        qs = urlparse(self.path).query
        if "format=json" in qs:
            fmt = "json"
        if kind == "rl" and rl:
            if fmt == "json":
                data = json.dumps(rl, cls=NumpyEncoder)
                ct, ext = "application/json", "json"
            else:
                data = traj_to_csv(rl)
                ct, ext = "text/csv", "csv"
        elif kind == "baseline" and bl:
            if fmt == "json":
                data = json.dumps(bl, cls=NumpyEncoder)
                ct, ext = "application/json", "json"
            else:
                data = traj_to_csv(bl)
                ct, ext = "text/csv", "csv"
        elif kind == "comparison" and rl and bl:
            rows = []
            for i, (r, b) in enumerate(zip(rl, bl)):
                rows.append({"hour": i, **{f"rl_{k}": v for k, v in r.items()}, **{f"bl_{k}": v for k, v in b.items()}})
            if fmt == "json":
                data = json.dumps(rows, cls=NumpyEncoder)
                ct, ext = "application/json", "json"
            else:
                buf = io.StringIO()
                if rows:
                    w = csv.DictWriter(buf, fieldnames=rows[0].keys())
                    w.writeheader()
                    w.writerows(rows)
                data = buf.getvalue()
                ct, ext = "text/csv", "csv"
        elif kind == "training":
            rw = self.st.get("rewards")
            cf = self.st.get("comforts")
            en = self.st.get("energies")
            rows = [{"episode": i, "reward": rw[i], "discomfort": cf[i], "energy": en[i]}
                    for i in range(len(rw))]
            if fmt == "json":
                data = json.dumps(rows, cls=NumpyEncoder)
                ct, ext = "application/json", "json"
            else:
                buf = io.StringIO()
                if rows:
                    w = csv.DictWriter(buf, fieldnames=rows[0].keys())
                    w.writeheader()
                    w.writerows(rows)
                data = buf.getvalue()
                ct, ext = "text/csv", "csv"
        else:
            return self._json({"error": "No data to export"}, 404)
        self.send_response(200)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Disposition", f'attachment; filename="hvac_{kind}.{ext}"')
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data.encode())

    def _train(self, cfg):
        if self.st.get("training"):
            return self._json({"error": "Already training"})
        algo = cfg.get("algo", "SAC")
        total = int(cfg.get("timesteps", 50000))
        ew = float(cfg.get("energy_weight", 0.6))
        lr = float(cfg.get("lr", 3e-4))
        timestep_min = int(cfg.get("timestep_min", 5))
        gamma = float(cfg.get("gamma", 0.99))
        tau = float(cfg.get("tau", 0.005))
        buffer_size = int(cfg.get("buffer_size", 50000))
        batch_size = int(cfg.get("batch_size", 256))
        continue_from = cfg.get("continue_from", None)
        prev_ts = 0
        if continue_from:
            model, meta = load_agent_model(continue_from)
            if model is None:
                return self._json({"error": f"Agent '{continue_from}' not found"})
            prev_ts = meta.get("train_cfg", {}).get("timesteps", 0)
            algo = meta.get("algo", algo)
        else:
            model = None
        self.st.update(training=True, algo=algo, total=prev_ts + total, current=prev_ts,
                       rewards=[], comforts=[], energies=[], status="training", error=None,
                       previous_timesteps=prev_ts, timestep_min=timestep_min)
        env_cfg = _env_kwargs(cfg)
        threading.Thread(target=self._worker, args=(algo, total, ew, lr, model, prev_ts, timestep_min, gamma, tau, buffer_size, batch_size, env_cfg), daemon=True).start()
        self._json({"ok": True, "continued_from": continue_from})

    def _worker(self, algo, total, ew, lr, existing_model=None, prev_ts=0, timestep_min=5,
                gamma=0.99, tau=0.005, buffer_size=50000, batch_size=256, env_cfg=None):
        try:
            weather = load_weather(timestep_min)
            env = Monitor(SimpleHVACEnv(weather, ew, timestep_min=timestep_min, **(env_cfg or {})))
            cb = CB(self.st)
            if existing_model is not None:
                m = existing_model
                m.set_env(env)
            elif algo == "SAC":
                m = SAC("MlpPolicy", env, learning_rate=lr, buffer_size=buffer_size,
                        batch_size=batch_size, gamma=gamma, tau=tau, verbose=0)
            elif algo == "PPO":
                m = PPO("MlpPolicy", env, learning_rate=lr, n_steps=256,
                        batch_size=batch_size, gamma=gamma, verbose=0)
            else:
                m = DDPG("MlpPolicy", env, learning_rate=lr, buffer_size=buffer_size,
                         batch_size=batch_size, gamma=gamma, tau=tau, verbose=0)
            m.learn(total_timesteps=total, callback=cb, reset_num_timesteps=existing_model is None)
            env.close()

            rw = [float(x) for x in self.st.get("rewards")]
            cf = [float(x) for x in self.st.get("comforts")]
            en = [float(x) for x in self.st.get("energies")]
            metrics = {
                "episode_rewards": rw, "episode_comforts": cf, "episode_energies": en,
                "avg_reward": sum(rw)/len(rw) if rw else 0,
                "avg_comfort": sum(cf)/len(cf) if cf else 0,
                "avg_energy": sum(en)/len(en) if en else 0,
            }
            total_ts = prev_ts + total
            train_cfg = {"algo": algo, "timesteps": total_ts, "energy_weight": ew, "lr": lr,
                         "timestep_min": timestep_min, "gamma": gamma, "tau": tau,
                         "buffer_size": buffer_size, "batch_size": batch_size}
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = f"{algo.lower()}_{ts}"
            save_agent(name, algo, m, metrics, train_cfg)
            self.st.update(model=m, status="done", training=False, last_saved=name)
        except Exception as e:
            self.st.update(status="error", error=str(e), training=False)

    def _simulate(self, cfg):
        model = self.st.get("model")
        if model is None:
            return self._json({"error": "No model trained yet"})
        ew = float(cfg.get("energy_weight", 0.6))
        timestep_min = int(cfg.get("timestep_min", 5))
        env_cfg = _env_kwargs(cfg)
        weather = load_weather(timestep_min)
        rl_traj = run_episode(model, weather, ew=ew, env_cfg=env_cfg, timestep_min=timestep_min)
        rl_m = sim_metrics(rl_traj)
        bl_traj = run_baseline(weather, ew=ew, env_cfg=env_cfg, timestep_min=timestep_min)
        bl_m = sim_metrics(bl_traj)
        self.st.update(last_rl_traj=rl_traj, last_bl_traj=bl_traj)
        self._json(dict(
            rl=dict(trajectory=rl_traj, metrics=rl_m),
            baseline=dict(trajectory=bl_traj, metrics=bl_m),
            hours=list(range(len(rl_traj)))))

    def _simulate_saved(self, cfg):
        name = cfg.get("name")
        if not name:
            return self._json({"error": "No agent name provided"})
        model, meta = load_agent_model(name)
        if model is None:
            return self._json({"error": f"Agent '{name}' not found"})
        ew = float(cfg.get("energy_weight", 0.6))
        timestep_min = int(cfg.get("timestep_min", 5))
        env_cfg = _env_kwargs(cfg)
        weather = load_weather(timestep_min)
        rl_traj = run_episode(model, weather, ew=ew, env_cfg=env_cfg, timestep_min=timestep_min)
        rl_m = sim_metrics(rl_traj)
        bl_traj = run_baseline(weather, ew=ew, env_cfg=env_cfg, timestep_min=timestep_min)
        bl_m = sim_metrics(bl_traj)
        self.st.update(last_rl_traj=rl_traj, last_bl_traj=bl_traj)
        self._json(dict(
            agent=meta, rl=dict(trajectory=rl_traj, metrics=rl_m),
            baseline=dict(trajectory=bl_traj, metrics=bl_m),
            hours=list(range(len(rl_traj)))))

    def _compare_agents(self, cfg):
        names = cfg.get("names", [])
        ew = float(cfg.get("energy_weight", 0.6))
        n = int(cfg.get("steps", 4032))
        env_cfg = _env_kwargs(cfg)
        if not names:
            return self._json({"error": "No agent names provided"})
        agents = []
        for name in names:
            model, meta = load_agent_model(name)
            if model is None:
                continue
            rl_traj = run_episode(model, self.weather, n, ew, env_cfg)
            rl_m = sim_metrics(rl_traj)
            agents.append(dict(name=name, meta=meta, trajectory=rl_traj, metrics=rl_m))
        bl_traj = run_baseline(self.weather, n, ew, env_cfg)
        bl_m = sim_metrics(bl_traj)
        self._json(dict(
            agents=agents,
            baseline=dict(trajectory=bl_traj, metrics=bl_m),
            hours=list(range(n))))

    def _sweep(self, cfg):
        if Handler.sweep_state and Handler.sweep_state.get("running"):
            return self._json({"error": "Sweep already running"})
        algo = cfg.get("algo", "SAC")
        base_ts = int(cfg.get("timesteps", 10000))
        ew = float(cfg.get("energy_weight", 0.6))
        timestep_min = int(cfg.get("timestep_min", 5))
        lrs = cfg.get("learning_rates", [1e-4, 3e-4, 1e-3])
        gammas = cfg.get("gammas", [0.99])
        results = []
        Handler.sweep_state = dict(running=True, total=len(lrs)*len(gammas), completed=0, results=[], error=None)

        def _sweep_worker():
            try:
                weather = load_weather(timestep_min)
                for lr in lrs:
                    for gamma in gammas:
                        env = Monitor(SimpleHVACEnv(weather, ew, timestep_min=timestep_min))
                        cb = CB(self.st)
                        if algo == "SAC":
                            m = SAC("MlpPolicy", env, learning_rate=lr, buffer_size=50000,
                                    batch_size=256, gamma=gamma, tau=0.005, verbose=0)
                        elif algo == "PPO":
                            m = PPO("MlpPolicy", env, learning_rate=lr, n_steps=256,
                                    batch_size=256, gamma=gamma, verbose=0)
                        else:
                            m = DDPG("MlpPolicy", env, learning_rate=lr, buffer_size=50000,
                                     batch_size=256, gamma=gamma, tau=0.005, verbose=0)
                        st_tmp = State()
                        cb2 = CB(st_tmp)
                        m.learn(total_timesteps=base_ts, callback=cb2)
                        env.close()
                        rw = [float(x) for x in st_tmp.get("rewards")]
                        cf = [float(x) for x in st_tmp.get("comforts")]
                        en = [float(x) for x in st_tmp.get("energies")]
                        metrics = {
                            "episode_rewards": rw, "episode_comforts": cf, "episode_energies": en,
                            "avg_reward": sum(rw)/len(rw) if rw else 0,
                            "avg_comfort": sum(cf)/len(cf) if cf else 0,
                            "avg_energy": sum(en)/len(en) if en else 0,
                        }
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        name = f"sweep_{algo.lower()}_lr{lr:.0e}_g{gamma}_{ts}"
                        save_agent(name, algo, m, metrics, {"algo": algo, "timesteps": base_ts, "energy_weight": ew, "lr": lr, "gamma": gamma, "timestep_min": timestep_min})
                        Handler.sweep_state["completed"] += 1
                        Handler.sweep_state["results"].append(dict(name=name, lr=lr, gamma=gamma, metrics=metrics))
                Handler.sweep_state["running"] = False
            except Exception as e:
                Handler.sweep_state["running"] = False
                Handler.sweep_state["error"] = str(e)

        threading.Thread(target=_sweep_worker, daemon=True).start()
        self._json({"ok": True, "total": len(lrs)*len(gammas)})


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    print("=" * 50)
    print("  HVAC RL Control Platform")
    print("=" * 50)
    weather = load_weather(5)
    print(f"Loaded {len(weather)} rows of weather data (5-min default)")
    st = State()
    bl = run_baseline(weather, ew=0.6, timestep_min=5)
    bl_m = sim_metrics(bl)
    bl_hours = list(range(len(bl)))
    bl_data = dict(trajectory=bl, metrics=bl_m, hours=bl_hours)
    print(f"Baseline: comfort={bl_m['comfort_score']:.1f}% energy={bl_m['total_energy']:.3f} kWh/m2")
    html = str(Path(__file__).parent / "dashboard.html")
    Handler.st = st
    Handler.weather = weather
    Handler.html_path = html
    Handler.baseline = bl_data
    srv = ThreadedHTTPServer(("0.0.0.0", port), Handler)
    print(f"\nOpen http://localhost:{port} in your browser\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nDone.")


if __name__ == "__main__":
    main()
