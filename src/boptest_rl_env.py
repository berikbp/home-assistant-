from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np
import requests
from gymnasium import spaces


class BoptestRLEnv(gym.Env):
    """
    Reconstructed RL environment for BOPTEST bestest_air.

    RL controls:
        1. FCU fan command
        2. FCU supply-air temperature

    Control interval:
        3600 s = 1 hour

    Important:
        This is a reconstructed methodology because the exact preliminary
        RL environment/reward implementation was not recovered.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        testcase: str = "bestest_air",
        energy_weight: float = 0.6,
        episode_steps: int = 24,
        start_time: int = 0,
        warmup_period: int = 0,
        step_seconds: int = 3600,
        request_timeout: int = 600,
    ):
        super().__init__()

        self.base_url = base_url.rstrip("/")
        self.testcase = testcase

        self.energy_weight = float(energy_weight)
        self.episode_steps = int(episode_steps)

        self.start_time = int(start_time)
        self.warmup_period = int(warmup_period)
        self.step_seconds = int(step_seconds)

        self.request_timeout = int(request_timeout)

        # bestest_air floor area from model documentation
        self.floor_area_m2 = 48.0

        # ------------------------------------------------------------
        # Reference PI baseline:
        # 14 days = 336 hourly steps
        # ------------------------------------------------------------
        self.baseline_tdis = 6.8246808964870205
        self.baseline_energy = 4.226250612177942

        self.discomfort_ref = self.baseline_tdis / 336.0
        self.energy_ref = self.baseline_energy / 336.0

        # ------------------------------------------------------------
        # SB3 normalized actions
        #
        # action[0] = fan
        # action[1] = supply-air temperature
        # ------------------------------------------------------------
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )

        # ------------------------------------------------------------
        # Observation:
        #
        # 0 zone temperature [C]
        # 1 outdoor temperature [C]
        # 2 solar [W/m2]
        # 3 lower comfort limit [C]
        # 4 upper comfort limit [C]
        # 5 occupancy
        # 6 sin(hour)
        # 7 cos(hour)
        # ------------------------------------------------------------
        self.observation_space = spaces.Box(
            low=np.array(
                [
                    -20.0,   # zone C
                    -50.0,   # outdoor C
                    0.0,     # solar
                    0.0,     # lower setpoint C
                    0.0,     # upper setpoint C
                    0.0,     # occupancy
                    -1.0,    # sin
                    -1.0,    # cos
                ],
                dtype=np.float32,
            ),
            high=np.array(
                [
                    60.0,
                    60.0,
                    1500.0,
                    50.0,
                    50.0,
                    20.0,
                    1.0,
                    1.0,
                ],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

        self.testid: str | None = None
        self.current_step = 0
        self.current_time = self.start_time

        self.schedule_time: np.ndarray | None = None
        self.schedule_lower: np.ndarray | None = None
        self.schedule_upper: np.ndarray | None = None
        self.schedule_occ: np.ndarray | None = None

        self.last_payload: dict[str, Any] | None = None

    # ================================================================
    # HTTP helpers
    # ================================================================

    def _request(
        self,
        method: str,
        endpoint: str,
        **kwargs,
    ) -> dict[str, Any]:

        response = requests.request(
            method,
            f"{self.base_url}{endpoint}",
            timeout=self.request_timeout,
            **kwargs,
        )

        response.raise_for_status()

        data = response.json()

        # Some BOPTEST endpoints wrap data under "payload".
        if isinstance(data, dict) and "payload" in data:
            return data["payload"]

        return data

    # ================================================================
    # Test lifecycle
    # ================================================================

    def _select_testcase(self) -> str:

        data = self._request(
            "POST",
            f"/testcases/{self.testcase}/select",
        )

        # Depending on BOPTEST API version, testid may be directly
        # returned or wrapped.
        if "testid" in data:
            return str(data["testid"])

        raise RuntimeError(
            f"BOPTEST testcase selection did not return testid: {data}"
        )

    def _stop_testcase(self):

        if self.testid is None:
            return

        try:
            requests.put(
                f"{self.base_url}/stop/{self.testid}",
                timeout=10,
            )
        except requests.RequestException:
            pass

        self.testid = None

    # ================================================================
    # Schedule
    # ================================================================

    def _load_schedule(self):

        if self.testid is None:
            raise RuntimeError("No active BOPTEST test.")

        horizon = self.episode_steps * self.step_seconds

        payload = {
            "point_names": [
                "LowerSetp[1]",
                "UpperSetp[1]",
                "Occupancy[1]",
            ],
            "horizon": horizon,
            "interval": 600,
        }

        schedule = self._request(
            "PUT",
            f"/forecast/{self.testid}",
            json=payload,
        )

        self.schedule_time = np.asarray(
            schedule["time"],
            dtype=np.float64,
        )

        self.schedule_lower = np.asarray(
            schedule["LowerSetp[1]"],
            dtype=np.float64,
        )

        self.schedule_upper = np.asarray(
            schedule["UpperSetp[1]"],
            dtype=np.float64,
        )

        self.schedule_occ = np.asarray(
            schedule["Occupancy[1]"],
            dtype=np.float64,
        )

    def _schedule_at(
        self,
        time_seconds: float,
    ) -> tuple[float, float, float]:

        if self.schedule_time is None:
            raise RuntimeError("Schedule has not been loaded.")

        index = int(
            np.searchsorted(
                self.schedule_time,
                time_seconds,
                side="right",
            )
            - 1
        )

        index = int(
            np.clip(
                index,
                0,
                len(self.schedule_time) - 1,
            )
        )

        lower_k = float(self.schedule_lower[index])
        upper_k = float(self.schedule_upper[index])
        occupancy = float(self.schedule_occ[index])

        return (
            lower_k - 273.15,
            upper_k - 273.15,
            occupancy,
        )

    # ================================================================
    # Action mapping
    # ================================================================

    @staticmethod
    def _map_action(
        action: np.ndarray,
    ) -> tuple[float, float]:

        action = np.asarray(
            action,
            dtype=np.float64,
        )

        action = np.clip(
            action,
            -1.0,
            1.0,
        )

        # [-1, +1] -> [0, 1]
        fan = (action[0] + 1.0) / 2.0

        # [-1, +1] -> [285.15, 313.15] K
        t_supply_k = (
            285.15
            + ((action[1] + 1.0) / 2.0)
            * (313.15 - 285.15)
        )

        # Explicit clipping avoids floating-point boundary warnings.
        fan = float(
            np.clip(
                fan,
                0.0,
                1.0,
            )
        )

        t_supply_k = float(
            np.clip(
                t_supply_k,
                285.15,
                313.15,
            )
        )

        return fan, t_supply_k

    # ================================================================
    # Observation
    # ================================================================

    def _make_observation(
        self,
        payload: dict[str, Any],
    ) -> np.ndarray:

        time_seconds = float(payload["time"])

        zone_c = (
            float(payload["zon_reaTRooAir_y"])
            - 273.15
        )

        outdoor_c = (
            float(payload["zon_weaSta_reaWeaTDryBul_y"])
            - 273.15
        )

        solar = float(
            payload["zon_weaSta_reaWeaHGloHor_y"]
        )

        lower_c, upper_c, occupancy = (
            self._schedule_at(time_seconds)
        )

        hour = (
            (time_seconds % 86400.0)
            / 3600.0
        )

        hour_angle = (
            2.0
            * math.pi
            * hour
            / 24.0
        )

        obs = np.array(
            [
                zone_c,
                outdoor_c,
                solar,
                lower_c,
                upper_c,
                occupancy,
                math.sin(hour_angle),
                math.cos(hour_angle),
            ],
            dtype=np.float32,
        )

        return obs

    # ================================================================
    # Reward
    # ================================================================

    def _calculate_reward(
        self,
        payload: dict[str, Any],
    ) -> tuple[float, dict[str, float]]:

        time_seconds = float(payload["time"])

        zone_c = (
            float(payload["zon_reaTRooAir_y"])
            - 273.15
        )

        lower_c, upper_c, _ = (
            self._schedule_at(time_seconds)
        )

        low_violation = max(
            lower_c - zone_c,
            0.0,
        )

        high_violation = max(
            zone_c - upper_c,
            0.0,
        )

        # With a one-hour control step, K violation is numerically
        # equivalent to the simple K*h endpoint approximation.
        discomfort = (
            low_violation
            + high_violation
        )

        p_heat_w = max(
            float(payload["fcu_reaPHea_y"]),
            0.0,
        )

        p_cool_w = max(
            float(payload["fcu_reaPCoo_y"]),
            0.0,
        )

        p_fan_w = max(
            float(payload["fcu_reaPFan_y"]),
            0.0,
        )

        total_power_w = (
            p_heat_w
            + p_cool_w
            + p_fan_w
        )

        duration_hours = (
            self.step_seconds / 3600.0
        )

        energy_kwh_m2 = (
            total_power_w
            / 1000.0
            * duration_hours
            / self.floor_area_m2
        )

        discomfort_normalized = (
            discomfort
            / self.discomfort_ref
        )

        energy_normalized = (
            energy_kwh_m2
            / self.energy_ref
        )

        reward = -(
            discomfort_normalized
            + self.energy_weight
            * energy_normalized
        )

        reward_info = {
            "zone_temperature_c": zone_c,
            "lower_setpoint_c": lower_c,
            "upper_setpoint_c": upper_c,
            "discomfort_proxy": discomfort,
            "power_heat_w": p_heat_w,
            "power_cool_w": p_cool_w,
            "power_fan_w": p_fan_w,
            "energy_proxy_kwh_m2": energy_kwh_m2,
            "discomfort_normalized": discomfort_normalized,
            "energy_normalized": energy_normalized,
        }

        return float(reward), reward_info

    # ================================================================
    # Gymnasium interface
    # ================================================================

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ):

        super().reset(seed=seed)

        # -----------------------------------------------
        # Reuse existing test instance if available.
        # Only create a new one on the very first reset.
        # -----------------------------------------------

        if self.testid is None:
            self.testid = self._select_testcase()
            print(f"BOPTEST test ID: {self.testid}")

        init_payload = {
            "start_time": self.start_time,
            "warmup_period": self.warmup_period,
        }

        payload = self._request(
            "PUT",
            f"/initialize/{self.testid}",
            json=init_payload,
        )

        self.current_step = 0
        self.current_time = float(payload["time"])

        self._load_schedule()

        self.last_payload = payload

        observation = self._make_observation(
            payload
        )

        info = {
            "testid": self.testid,
            "time": self.current_time,
        }

        return observation, info

    def step(
        self,
        action: np.ndarray,
    ):

        if self.testid is None:
            raise RuntimeError(
                "Environment must be reset before step()."
            )

        fan, t_supply_k = self._map_action(
            action
        )

        command = {
            "fcu_oveFan_activate": 1,
            "fcu_oveFan_u": fan,

            "fcu_oveTSup_activate": 1,
            "fcu_oveTSup_u": t_supply_k,
        }

        payload = self._request(
            "POST",
            f"/advance/{self.testid}",
            json=command,
        )

        self.current_step += 1
        self.current_time = float(
            payload["time"]
        )

        observation = self._make_observation(
            payload
        )

        reward, reward_info = (
            self._calculate_reward(payload)
        )

        terminated = False

        truncated = (
            self.current_step
            >= self.episode_steps
        )

        info = {
            "testid": self.testid,
            "time": self.current_time,
            "step": self.current_step,
            "fan_command": fan,
            "supply_temperature_k": t_supply_k,
            **reward_info,
        }

        self.last_payload = payload

        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )

    def close(self):

        self._stop_testcase()
