#!/usr/bin/env python3
"""Inference wrapper: takes sensor readings, returns HVAC commands.

Usage:
    wrapper = HVACInference("sac_30k_fresh")
    action = wrapper.predict(outdoor_temp=25.0, solar=400.0, indoor_temp=22.0,
                             is_occupied=True, hour=14)
    # returns: {"fan_speed": 0.73, "supply_temp": 28.5}
"""

import sys, os, math, json
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from rl_platform import load_agent_model


class HVACInference:
    def __init__(self, model_name, models_dir=None):
        if models_dir is None:
            models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
        self.model, self.meta = load_agent_model(model_name)
        if self.model is None:
            raise FileNotFoundError(f"Model '{model_name}' not found in {models_dir}")

        cfg = self.meta.get("train_cfg", {})
        self.occ_lo = cfg.get("occ_setpoint_lower", 21)
        self.occ_hi = cfg.get("occ_setpoint_upper", 24)
        self.unocc_lo = cfg.get("unocc_setpoint_lower", 15)
        self.unocc_hi = cfg.get("unocc_setpoint_upper", 30)
        self.occ_hours = cfg.get("occ_hours", list(range(8, 20)))

    def _build_obs(self, outdoor_temp, solar, indoor_temp, is_occupied, hour):
        if is_occupied:
            lo, hi = self.occ_lo, self.occ_hi
            occ = 1
        else:
            lo, hi = self.unocc_lo, self.unocc_hi
            occ = 0

        a = 2 * math.pi * (hour % 24) / 24
        return np.array([indoor_temp, outdoor_temp, solar, lo, hi, occ,
                         math.sin(a), math.cos(a)], np.float32)

    def predict(self, outdoor_temp, solar, indoor_temp, is_occupied, hour):
        obs = self._build_obs(outdoor_temp, solar, indoor_temp, is_occupied, hour)
        action, _ = self.model.predict(obs, deterministic=True)
        action = np.clip(action, -1, 1)
        fan = float((action[0] + 1) / 2)
        tsup = float(12 + (action[1] + 1) / 2 * 28)
        return {"fan_speed": round(fan, 4), "supply_temp": round(tsup, 2)}

    def predict_batch(self, readings):
        results = []
        for r in readings:
            results.append(self.predict(
                outdoor_temp=r["outdoor_temp"],
                solar=r.get("solar", 0),
                indoor_temp=r["indoor_temp"],
                is_occupied=r.get("is_occupied", False),
                hour=r.get("hour", 12),
            ))
        return results


def test_wrapper():
    model_name = "sac_30k_fresh"
    print(f"Loading model: {model_name}")
    w = HVACInference(model_name)
    print(f"Model: {w.meta['algo']}, {w.meta['train_cfg']['timesteps']} steps")

    test_cases = [
        {"outdoor_temp": 30, "solar": 600, "indoor_temp": 23, "is_occupied": True, "hour": 14},
        {"outdoor_temp": 5, "solar": 0, "indoor_temp": 18, "is_occupied": True, "hour": 9},
        {"outdoor_temp": 20, "solar": 0, "indoor_temp": 22, "is_occupied": False, "hour": 23},
        {"outdoor_temp": 35, "solar": 800, "indoor_temp": 25, "is_occupied": True, "hour": 12},
    ]

    for tc in test_cases:
        result = w.predict(**tc)
        print(f"\n  Input: T_out={tc['outdoor_temp']}C, G={tc['solar']}W/m2, "
              f"T_in={tc['indoor_temp']}C, occ={tc['is_occupied']}, h={tc['hour']}")
        print(f"  Output: fan={result['fan_speed']:.2f}, T_sup={result['supply_temp']:.1f}C")

    print("\nAll tests passed.")


if __name__ == "__main__":
    test_wrapper()
