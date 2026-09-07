import numpy as np
import gymnasium as gym
from gymnasium import spaces
import requests


class BoptestDebugEnv(gym.Env):
    """
    Debug-only BOPTEST Gymnasium environment.

    IMPORTANT:
    This is NOT yet the final RL environment used in the paper.

    Observation:
        [
            room_temperature_C,
            outdoor_temperature_C,
            global_solar_radiation_W_m2
        ]

    Agent action:
        one normalized continuous value in [-1, 1]

    Physical actuator:
        fan command in [0, 1]

    Reward:
        0.0

    The zero reward is intentional. We have not yet implemented
    the preliminary study's actual comfort-energy reward.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        base_url="http://127.0.0.1:8000",
        max_steps=24,
    ):
        super().__init__()

        self.base_url = base_url

        self.testid = None

        self.current_step = 0
        self.max_steps = max_steps

        # ----------------------------------------------------
        # Agent action space
        #
        # Stable-Baselines3 recommends normalized symmetric
        # continuous actions.
        #
        # RL agent:
        #
        #       [-1, +1]
        #
        # which we internally map to physical fan command:
        #
        #       [0, 1]
        # ----------------------------------------------------

        self.action_space = spaces.Box(
            low=np.array(
                [-1.0],
                dtype=np.float32,
            ),
            high=np.array(
                [1.0],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

        # ----------------------------------------------------
        # Debug observation space
        #
        # These limits are broad engineering bounds.
        # This is not yet the final observation definition.
        # ----------------------------------------------------

        self.observation_space = spaces.Box(
            low=np.array(
                [
                    -50.0,   # room temperature [°C]
                    -50.0,   # outdoor temperature [°C]
                    0.0,     # solar radiation [W/m²]
                ],
                dtype=np.float32,
            ),
            high=np.array(
                [
                    80.0,
                    80.0,
                    1500.0,
                ],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

    # ========================================================
    # HTTP helper
    # ========================================================

    def _request(
        self,
        method,
        endpoint,
        **kwargs,
    ):
        response = requests.request(
            method,
            f"{self.base_url}/{endpoint}",
            timeout=120,
            **kwargs,
        )

        response.raise_for_status()

        result = response.json()

        if isinstance(result, dict):

            status = result.get("status")

            if (
                status is not None
                and status != 200
            ):
                raise RuntimeError(
                    f"BOPTEST request failed: {result}"
                )

        return result

    # ========================================================
    # Test-case lifecycle
    # ========================================================

    def _select_testcase(self):

        result = self._request(
            "POST",
            "testcases/bestest_air/select",
        )

        self.testid = result["testid"]

    def _stop_testcase(self):

        if self.testid is None:
            return

        try:

            requests.put(
                f"{self.base_url}/stop/{self.testid}",
                timeout=30,
            )

        except Exception:
            pass

        self.testid = None

    # ========================================================
    # Observation construction
    # ========================================================

    def _make_observation(
        self,
        data,
    ):

        room_temperature_C = (
            data["zon_reaTRooAir_y"]
            - 273.15
        )

        outdoor_temperature_C = (
            data["zon_weaSta_reaWeaTDryBul_y"]
            - 273.15
        )

        solar_radiation = (
            data[
                "zon_weaSta_reaWeaHGloHor_y"
            ]
        )

        observation = np.array(
            [
                room_temperature_C,
                outdoor_temperature_C,
                solar_radiation,
            ],
            dtype=np.float32,
        )

        return observation

    # ========================================================
    # Action mapping
    # ========================================================

    @staticmethod
    def _map_agent_action_to_fan(
        agent_action,
    ):
        """
        Map:

            agent action [-1, 1]

        to:

            BOPTEST fan command [0, 1]
        """

        agent_action = float(
            np.clip(
                agent_action,
                -1.0,
                1.0,
            )
        )

        fan_command = (
            agent_action + 1.0
        ) / 2.0

        return fan_command

    # ========================================================
    # Gymnasium reset
    # ========================================================

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):
        super().reset(
            seed=seed
        )

        # -----------------------------------------------
        # Release any previous BOPTEST instance
        # -----------------------------------------------

        self._stop_testcase()

        # -----------------------------------------------
        # Start fresh bestest_air instance
        # -----------------------------------------------

        self._select_testcase()

        # -----------------------------------------------
        # Initialize simulation.
        #
        # IMPORTANT:
        # /initialize already returns the current state.
        #
        # Therefore reset() does NOT advance one hour.
        # -----------------------------------------------

        result = self._request(
            "PUT",
            f"initialize/{self.testid}",
            json={
                "start_time": 0,
                "warmup_period": 0,
            },
        )

        data = result["payload"]

        self.current_step = 0

        observation = (
            self._make_observation(
                data
            )
        )

        info = {
            "time": data["time"],
            "current_step":
                self.current_step,
            "testid":
                self.testid,
        }

        return observation, info

    # ========================================================
    # Gymnasium step
    # ========================================================

    def step(
        self,
        action,
    ):

        # -----------------------------------------------
        # 1. Read normalized RL action
        # -----------------------------------------------

        agent_action = float(
            action[0]
        )

        # -----------------------------------------------
        # 2. Convert to physical fan command
        # -----------------------------------------------

        fan_command = (
            self._map_agent_action_to_fan(
                agent_action
            )
        )

        # -----------------------------------------------
        # 3. Send actual control to BOPTEST
        # -----------------------------------------------

        payload = {
            "fcu_oveFan_activate": 1,
            "fcu_oveFan_u":
                fan_command,
        }

        result = self._request(
            "POST",
            f"advance/{self.testid}",
            json=payload,
        )

        data = result["payload"]

        self.current_step += 1

        # -----------------------------------------------
        # 4. New state
        # -----------------------------------------------

        observation = (
            self._make_observation(
                data
            )
        )

        # -----------------------------------------------
        # 5. Reward
        #
        # DEBUG ONLY.
        #
        # We deliberately do NOT invent the final
        # comfort-energy reward.
        # -----------------------------------------------

        reward = 0.0

        # BOPTEST itself does not define an RL terminal
        # state for this debug episode.
        terminated = False

        # -----------------------------------------------
        # 6. Episode-length truncation
        #
        # 24 steps now means exactly 24 AGENT ACTIONS.
        # -----------------------------------------------

        truncated = (
            self.current_step
            >= self.max_steps
        )

        # -----------------------------------------------
        # 7. Diagnostics
        # -----------------------------------------------

        info = {

            "time":
                data["time"],

            "current_step":
                self.current_step,

            "agent_action":
                agent_action,

            "fan_command":
                fan_command,

            "fan_power_W":
                data[
                    "fcu_reaPFan_y"
                ],

            "heating_power_W":
                data[
                    "fcu_reaPHea_y"
                ],

            "cooling_power_W":
                data[
                    "fcu_reaPCoo_y"
                ],

            "supply_air_flow_kg_s":
                data[
                    "fcu_reaFloSup_y"
                ],

            "room_temperature_C":
                data[
                    "zon_reaTRooAir_y"
                ]
                - 273.15,

            "outdoor_temperature_C":
                data[
                    "zon_weaSta_reaWeaTDryBul_y"
                ]
                - 273.15,
        }

        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )

    # ========================================================
    # Cleanup
    # ========================================================

    def close(self):

        self._stop_testcase()