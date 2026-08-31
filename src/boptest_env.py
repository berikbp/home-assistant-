import numpy as np
import gymnasium as gym
from gymnasium import spaces
import requests


class BoptestDebugEnv(gym.Env):
    """
    Debug-only Gymnasium environment.

    IMPORTANT:
    This is NOT yet the final paper environment.

    Observation:
        [room_temperature_C,
         outdoor_temperature_C,
         global_solar_W_m2]

    Action:
        fan command in [0, 1]

    Reward:
        0.0 for now.

    We intentionally do NOT invent the final reward.
    """

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        self.base_url = "http://127.0.0.1:8000"
        self.testid = None

        # One continuous action:
        # normalized fan command from 0 to 1.
        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Observation:
        # room temp C,
        # outdoor temp C,
        # global solar radiation
        self.observation_space = spaces.Box(
            low=np.array(
                [-50.0, -50.0, 0.0],
                dtype=np.float32,
            ),
            high=np.array(
                [80.0, 80.0, 1500.0],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

        self.current_step = 0
        self.max_steps = 24

    def _select_testcase(self):
        response = requests.post(
            f"{self.base_url}/testcases/bestest_air/select",
            timeout=60,
        )

        response.raise_for_status()

        self.testid = response.json()["testid"]

    def _initialize(self):
        response = requests.put(
            f"{self.base_url}/initialize/{self.testid}",
            json={
                "start_time": 0,
                "warmup_period": 0,
            },
            timeout=60,
        )

        response.raise_for_status()

    def _make_observation(self, data):

        room_c = (
            data["zon_reaTRooAir_y"]
            - 273.15
        )

        outdoor_c = (
            data["zon_weaSta_reaWeaTDryBul_y"]
            - 273.15
        )

        solar = (
            data["zon_weaSta_reaWeaHGloHor_y"]
        )

        obs = np.array(
            [
                room_c,
                outdoor_c,
                solar,
            ],
            dtype=np.float32,
        )

        return obs

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):
        super().reset(seed=seed)

        # Stop old test if one exists.
        if self.testid is not None:
            try:
                requests.put(
                    f"{self.base_url}/stop/{self.testid}",
                    timeout=30,
                )
            except Exception:
                pass

        self._select_testcase()
        self._initialize()

        self.current_step = 0

        # Advance once using built-in control
        # so that we obtain an observation.
        response = requests.post(
            f"{self.base_url}/advance/{self.testid}",
            json={},
            timeout=60,
        )

        response.raise_for_status()

        data = response.json()["payload"]

        self.current_step += 1

        observation = self._make_observation(data)

        info = {
            "time": data["time"],
        }

        return observation, info

    def step(self, action):

        fan_command = float(action[0])

        payload = {
            "fcu_oveFan_activate": 1,
            "fcu_oveFan_u": fan_command,
        }

        response = requests.post(
            f"{self.base_url}/advance/{self.testid}",
            json=payload,
            timeout=60,
        )

        response.raise_for_status()

        data = response.json()["payload"]

        self.current_step += 1

        observation = self._make_observation(data)

        # ----------------------------------------------------
        # DEBUG REWARD ONLY
        #
        # Do NOT use this in final experiments.
        #
        # We return zero because the preliminary paper reward
        # has not yet been supplied.
        # ----------------------------------------------------

        reward = 0.0

        terminated = False

        truncated = (
            self.current_step >= self.max_steps
        )

        info = {
            "time": data["time"],
            "fan_command": fan_command,
            "fan_power_W": data["fcu_reaPFan_y"],
            "heating_power_W": data["fcu_reaPHea_y"],
            "cooling_power_W": data["fcu_reaPCoo_y"],
            "room_temperature_C":
                data["zon_reaTRooAir_y"] - 273.15,
        }

        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )

    def close(self):

        if self.testid is not None:

            try:
                requests.put(
                    f"{self.base_url}/stop/{self.testid}",
                    timeout=30,
                )

            except Exception:
                pass

            self.testid = None
