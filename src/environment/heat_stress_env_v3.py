"""
Heat Stress Environment V3 - Designed for RL Advantage
========================================================
Key features that favor RL over simple thresholds:
1. Delayed cooling effects (takes time to work)
2. Individual cow sensitivity (varies per episode)
3. Cooling fatigue (repeated cooling less effective)
4. Weather forecasting reward (proactive planning)
5. Energy cost varies by time of day
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any


class HeatStressEnvV3(gym.Env):
    """
    Advanced Heat Stress Environment with features that require
    temporal reasoning and learning beyond simple thresholds.
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        episode_length: int = 48,
        render_mode: Optional[str] = None
    ):
        super().__init__()

        self.episode_length = episode_length
        self.render_mode = render_mode

        # Extended state space:
        # [THI, CBT, lying, activity, hour, milk, heat_load,
        #  prev_action, cow_sensitivity, cooling_fatigue, thi_trend]
        self.observation_space = spaces.Box(
            low=np.array([50, 38, 0, 0, 0, 0, 0, 0, 0.5, 0, -1]),
            high=np.array([90, 41, 1, 1, 23, 50, 100, 3, 1.5, 1, 1]),
            dtype=np.float32
        )

        self.action_space = spaces.Discrete(4)

        # Base costs (modified by time of day)
        self.base_energy_costs = {0: 0.0, 1: 0.3, 2: 1.0, 3: 0.5}
        self.base_cooling_effects = {0: 0.0, 1: 0.4, 2: 1.0, 3: 0.5}

        # Episode variables
        self.current_step = 0
        self.state = None
        self.cow_sensitivity = 1.0
        self.cooling_fatigue = 0.0
        self.prev_actions = []
        self.pending_cooling = []  # Delayed cooling effects

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        self.current_step = 0
        self.cooling_fatigue = 0.0
        self.prev_actions = []
        self.pending_cooling = []

        # Random cow sensitivity (some cows more heat-sensitive)
        self.cow_sensitivity = 0.7 + self.np_random.random() * 0.6  # 0.7 to 1.3

        # Initial conditions
        hour = self.np_random.integers(6, 12)
        thi = 62 + self.np_random.normal(0, 3)
        cbt = 38.3 + self.np_random.normal(0, 0.15)

        # Calculate THI trend (will THI increase or decrease?)
        thi_trend = 0.5 if hour < 14 else -0.3  # Rising in morning, falling in evening

        self.state = np.array([
            np.clip(thi, 50, 90),
            np.clip(cbt, 38.0, 41.0),
            0.5 + self.np_random.normal(0, 0.1),  # lying
            0.4 + self.np_random.normal(0, 0.1),  # activity
            hour,
            28 + self.np_random.normal(0, 2),  # milk
            0.0,  # heat_load
            0.0,  # prev_action
            self.cow_sensitivity,
            self.cooling_fatigue,
            thi_trend
        ], dtype=np.float32)

        return self.state, {"cow_sensitivity": self.cow_sensitivity}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        assert self.action_space.contains(action)

        thi, cbt, lying, activity, hour, milk, heat_load, prev_act, sens, fatigue, thi_trend = self.state

        # Store action for fatigue calculation
        self.prev_actions.append(action)
        if len(self.prev_actions) > 6:
            self.prev_actions.pop(0)

        # Calculate cooling fatigue (repeated cooling becomes less effective)
        recent_cooling = sum(1 for a in self.prev_actions if a > 0)
        self.cooling_fatigue = min(0.8, recent_cooling * 0.1)

        # Energy cost varies by time of day (peak hours more expensive)
        time_multiplier = 1.5 if 12 <= hour <= 18 else 1.0
        energy_cost = self.base_energy_costs[action] * time_multiplier

        # Cooling effect with delay and fatigue
        base_cooling = self.base_cooling_effects[action]
        effective_cooling = base_cooling * (1 - self.cooling_fatigue * 0.5)

        # Add pending cooling effect (delayed by 1 step)
        self.pending_cooling.append(effective_cooling * 0.3)  # 30% delayed
        immediate_cooling = effective_cooling * 0.7  # 70% immediate

        # Apply pending cooling from previous step
        delayed_effect = self.pending_cooling.pop(0) if self.pending_cooling else 0

        total_cooling = immediate_cooling + delayed_effect

        # CBT dynamics with cow sensitivity
        heat_gain = 0.03 * max(0, thi - 68) * self.cow_sensitivity
        cbt_new = cbt + heat_gain - total_cooling * 0.8
        cbt_new = np.clip(cbt_new + self.np_random.normal(0, 0.05), 38.0, 41.0)

        # Calculate stress level
        stress_level = self._get_stress_level(thi, cbt_new)

        # Update heat load
        heat_increment = max(0, (thi - 68) * 0.1 + (cbt_new - 38.5) * 0.5) * self.cow_sensitivity
        heat_reduction = total_cooling * 1.5
        new_heat_load = max(0, heat_load + heat_increment - heat_reduction)

        # Calculate reward
        reward = self._calculate_reward(
            thi, cbt, cbt_new, action, stress_level,
            energy_cost, new_heat_load, thi_trend
        )

        # Transition to next state
        self.current_step += 1
        next_state = self._transition(action, cbt_new, new_heat_load)
        self.state = next_state

        terminated = self.current_step >= self.episode_length
        info = {
            "stress_level": stress_level,
            "cooling_fatigue": self.cooling_fatigue,
            "cow_sensitivity": self.cow_sensitivity,
            "energy_cost": energy_cost
        }

        return next_state, reward, terminated, False, info

    def _get_stress_level(self, thi: float, cbt: float) -> int:
        thi_stress = 0 if thi < 68 else (1 if thi < 72 else (2 if thi < 79 else 3))
        cbt_stress = 0 if cbt < 39.0 else (1 if cbt < 39.5 else (2 if cbt < 40.0 else 3))
        return max(thi_stress, cbt_stress)

    def _calculate_reward(
        self, thi, cbt_old, cbt_new, action, stress_level,
        energy_cost, heat_load, thi_trend
    ) -> float:
        reward = 0.0

        # 1. Comfort reward (base)
        comfort = 1.0 / (1.0 + np.exp(0.3 * (cbt_new - 38.8)))
        reward += comfort * 1.5

        # 2. Stress penalty (progressive)
        stress_penalties = [0, -1.0, -2.5, -5.0]
        reward += stress_penalties[stress_level]

        # 3. Energy cost
        reward -= energy_cost * 0.5

        # 4. Proactive bonus: cooling when THI trending up
        if action > 0 and thi_trend > 0 and stress_level == 0:
            reward += 0.8  # Bonus for anticipatory cooling

        # 5. Recovery bonus
        if cbt_old > 39.0 and cbt_new < cbt_old:
            reward += (cbt_old - cbt_new) * 2.0

        # 6. Heat load penalty (cumulative stress is bad)
        reward -= heat_load * 0.02

        # 7. Efficiency bonus: mild cooling when appropriate
        if action == 1 and 0 < stress_level < 2:
            reward += 0.3  # Efficient use of mild cooling

        # 8. Penalty for cooling fatigue (over-cooling)
        if self.cooling_fatigue > 0.5 and action > 0:
            reward -= 0.5

        return reward

    def _transition(self, action: int, cbt_current: float, heat_load: float) -> np.ndarray:
        thi, _, lying, activity, hour, milk, _, _, sens, _, _ = self.state

        # Next hour
        hour_new = (hour + 1) % 24

        # THI with realistic daily pattern + some persistence
        base_thi = 68 + 12 * np.sin(np.pi * (hour_new - 5) / 12) if 5 <= hour_new <= 17 else 60
        thi_new = 0.7 * thi + 0.3 * base_thi + self.np_random.normal(0, 2)
        thi_new = np.clip(thi_new, 50, 90)

        # THI trend for next step
        next_base = 68 + 12 * np.sin(np.pi * ((hour_new + 1) % 24 - 5) / 12) if 5 <= (hour_new + 1) % 24 <= 17 else 60
        thi_trend = np.clip((next_base - base_thi) / 5, -1, 1)

        # CBT with inertia
        stress = self._get_stress_level(thi_new, cbt_current)
        cbt_new = cbt_current + 0.02 * max(0, thi_new - 68) * sens
        cbt_new = np.clip(cbt_new + self.np_random.normal(0, 0.05), 38.0, 41.0)

        # Lying affected by stress
        lying_new = 0.6 - 0.08 * stress + self.np_random.normal(0, 0.05)

        # Activity
        activity_new = 0.35 + self.np_random.normal(0, 0.05)

        # Milk affected by cumulative stress
        milk_new = milk * (1 - 0.001 * heat_load) + self.np_random.normal(0, 0.3)

        return np.array([
            np.clip(thi_new, 50, 90),
            cbt_new,
            np.clip(lying_new, 0, 1),
            np.clip(activity_new, 0, 1),
            hour_new,
            np.clip(milk_new, 10, 50),
            heat_load,
            float(action),
            self.cow_sensitivity,
            self.cooling_fatigue,
            thi_trend
        ], dtype=np.float32)


# Register
gym.register(
    id="HeatStressV3-v0",
    entry_point="src.environment.heat_stress_env_v3:HeatStressEnvV3",
)
