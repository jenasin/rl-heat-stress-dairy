"""
Heat Stress Environment V2 - Improved Reward Shaping
=====================================================
Key improvements:
1. Progressive heat stress penalties
2. Proactive cooling bonuses
3. Cumulative heat load tracking
4. Action consistency rewards
5. Better state representation
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any


class HeatStressEnvV2(gym.Env):
    """
    Improved Heat Stress Environment with better reward shaping.

    Key changes from V1:
    - Added cumulative heat load tracking
    - Progressive penalties for prolonged stress
    - Bonuses for proactive (preventive) cooling
    - Smoother reward landscape
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        data=None,
        reward_config: Dict[str, float] = None,
        episode_length: int = 48,
        render_mode: Optional[str] = None
    ):
        super().__init__()

        self.data = data
        self.episode_length = episode_length
        self.render_mode = render_mode

        # Improved reward configuration
        self.reward_config = reward_config or {
            # Base weights
            "milk_weight": 1.0,
            "comfort_weight": 0.8,
            "energy_penalty": 0.2,
            # New components
            "heat_stress_penalty": 2.0,      # Penalty for being in heat stress
            "proactive_bonus": 0.5,           # Bonus for cooling before stress
            "recovery_bonus": 1.0,            # Bonus for reducing CBT from stress
            "consistency_bonus": 0.1,         # Small bonus for consistent policy
            "cumulative_penalty": 0.05,       # Penalty per unit of accumulated heat load
        }

        # State space: [THI, CBT, lying, activity, hour, milk, heat_load, prev_action]
        # Added: heat_load (cumulative), prev_action (one-hot or scalar)
        self.observation_space = spaces.Box(
            low=np.array([50.0, 38.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            high=np.array([90.0, 41.0, 1.0, 1.0, 23.0, 50.0, 100.0, 3.0]),
            dtype=np.float32
        )

        # Action space: 4 discrete cooling interventions
        self.action_space = spaces.Discrete(4)

        # Action properties
        self.energy_costs = {0: 0.0, 1: 0.3, 2: 1.0, 3: 0.5}
        self.cooling_effects = {0: 0.0, 1: 0.4, 2: 1.0, 3: 0.5}

        # Episode tracking
        self.current_step = 0
        self.current_state = None
        self.prev_action = 0
        self.cumulative_heat_load = 0.0
        self.consecutive_stress_steps = 0
        self.episode_rewards = []

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        self.current_step = 0
        self.prev_action = 0
        self.cumulative_heat_load = 0.0
        self.consecutive_stress_steps = 0
        self.episode_rewards = []

        # Generate initial state
        self.current_state = self._generate_initial_state()

        info = {"step": 0, "heat_load": 0.0}
        return self.current_state.astype(np.float32), info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        assert self.action_space.contains(action)

        # Unpack state
        thi, cbt, lying, activity, hour, milk_prev, heat_load, _ = self.current_state

        # Determine stress level BEFORE action
        stress_level = self._get_stress_level(thi, cbt)
        was_stressed = stress_level > 0

        # Apply cooling effect
        cooling = self.cooling_effects[action]
        cbt_reduction = cooling * 0.8 * (1 + 0.3 * stress_level)  # More effective when stressed
        cbt_new = max(38.0, cbt - cbt_reduction)

        # Update cumulative heat load
        heat_increment = max(0, (thi - 68) * 0.1 + (cbt - 38.5) * 0.5)
        heat_reduction = cooling * 2.0
        self.cumulative_heat_load = max(0, self.cumulative_heat_load + heat_increment - heat_reduction)

        # Track consecutive stress
        if stress_level > 0:
            self.consecutive_stress_steps += 1
        else:
            self.consecutive_stress_steps = 0

        # Calculate reward components
        reward, reward_info = self._calculate_reward(
            thi, cbt, cbt_new, lying, action, stress_level, was_stressed
        )

        # Transition to next state
        self.current_step += 1
        next_state = self._transition_state(action, cbt_new)
        self.prev_action = action
        self.current_state = next_state
        self.episode_rewards.append(reward)

        # Termination
        terminated = self.current_step >= self.episode_length
        truncated = False

        info = {
            "step": self.current_step,
            "thi": thi,
            "cbt": cbt_new,
            "stress_level": stress_level,
            "heat_load": self.cumulative_heat_load,
            "action": action,
            "reward_breakdown": reward_info
        }

        return next_state.astype(np.float32), reward, terminated, truncated, info

    def _get_stress_level(self, thi: float, cbt: float) -> int:
        """Get heat stress level: 0=none, 1=mild, 2=moderate, 3=severe."""
        thi_level = 0 if thi < 68 else (1 if thi < 72 else (2 if thi < 79 else 3))
        cbt_level = 0 if cbt < 39.0 else (1 if cbt < 39.5 else (2 if cbt < 40.0 else 3))
        return max(thi_level, cbt_level)

    def _calculate_reward(
        self, thi: float, cbt: float, cbt_new: float,
        lying: float, action: int, stress_level: int, was_stressed: bool
    ) -> Tuple[float, Dict]:
        """Calculate reward with improved shaping."""
        cfg = self.reward_config
        reward_info = {}

        # 1. Comfort reward (base)
        comfort = self._calculate_comfort(thi, cbt_new, lying)
        comfort_reward = cfg["comfort_weight"] * comfort
        reward_info["comfort"] = comfort_reward

        # 2. Energy cost
        energy_cost = cfg["energy_penalty"] * self.energy_costs[action]
        reward_info["energy"] = -energy_cost

        # 3. Heat stress penalty (progressive)
        stress_penalty = 0.0
        if stress_level > 0:
            # Progressive penalty: mild=1, moderate=2, severe=4
            stress_multiplier = [0, 1, 2, 4][stress_level]
            stress_penalty = cfg["heat_stress_penalty"] * stress_multiplier
            # Extra penalty for prolonged stress
            stress_penalty += cfg["cumulative_penalty"] * self.consecutive_stress_steps
        reward_info["stress_penalty"] = -stress_penalty

        # 4. Proactive cooling bonus
        proactive_bonus = 0.0
        if not was_stressed and action > 0 and thi > 65:
            # Reward for cooling before stress hits
            proactive_bonus = cfg["proactive_bonus"] * (thi - 65) / 10
        reward_info["proactive"] = proactive_bonus

        # 5. Recovery bonus (for reducing CBT when stressed)
        recovery_bonus = 0.0
        if was_stressed and cbt_new < cbt:
            recovery_bonus = cfg["recovery_bonus"] * (cbt - cbt_new)
        reward_info["recovery"] = recovery_bonus

        # 6. Cumulative heat load penalty
        heat_load_penalty = cfg["cumulative_penalty"] * self.cumulative_heat_load
        reward_info["heat_load"] = -heat_load_penalty

        # 7. Milk production proxy (based on comfort and stress)
        milk_factor = comfort * (1 - 0.1 * stress_level)
        milk_reward = cfg["milk_weight"] * milk_factor
        reward_info["milk"] = milk_reward

        # Total reward
        total_reward = (
            comfort_reward
            - energy_cost
            - stress_penalty
            + proactive_bonus
            + recovery_bonus
            - heat_load_penalty
            + milk_reward
        )

        return total_reward, reward_info

    def _calculate_comfort(self, thi: float, cbt: float, lying: float) -> float:
        """Calculate comfort score with smoother curves."""
        # THI component - sigmoid-like curve
        thi_excess = max(0, thi - 68)
        thi_score = 1.0 / (1.0 + np.exp(0.2 * (thi_excess - 10)))

        # CBT component - Gaussian-like around optimal
        cbt_score = np.exp(-((cbt - 38.5) ** 2) / 2.0)

        # Lying - normal range
        lying_score = 1.0 - min(1.0, abs(lying - 0.55) * 2)

        return 0.4 * thi_score + 0.45 * cbt_score + 0.15 * lying_score

    def _generate_initial_state(self) -> np.ndarray:
        """Generate initial state with realistic starting conditions."""
        hour = self.np_random.integers(6, 18)  # Start during day

        # Moderate initial THI
        thi = 65 + self.np_random.normal(0, 5)
        thi = np.clip(thi, 55, 80)

        # Normal CBT
        cbt = 38.5 + self.np_random.normal(0, 0.2)
        cbt = np.clip(cbt, 38.2, 39.2)

        lying = 0.5 + self.np_random.normal(0, 0.1)
        activity = 0.4 + self.np_random.normal(0, 0.1)
        milk = 30 + self.np_random.normal(0, 3)

        return np.array([
            np.clip(thi, 50, 90),
            np.clip(cbt, 38.0, 41.0),
            np.clip(lying, 0, 1),
            np.clip(activity, 0, 1),
            hour,
            np.clip(milk, 15, 45),
            0.0,  # initial heat load
            0.0   # no previous action
        ])

    def _transition_state(self, action: int, cbt_current: float) -> np.ndarray:
        """Transition to next state with more realistic dynamics."""
        thi, cbt, lying, activity, hour, milk, heat_load, _ = self.current_state

        # Next hour
        hour_new = (hour + 1) % 24

        # THI follows realistic daily pattern
        # Peak at 14:00, minimum at 05:00
        hour_factor = np.sin(np.pi * (hour_new - 5) / 12) if 5 <= hour_new <= 17 else -0.3
        thi_base = 68 + 12 * max(0, hour_factor)
        thi_new = thi_base + self.np_random.normal(0, 3)

        # CBT dynamics - affected by THI and cooling persistence
        thi_effect = 0.03 * max(0, thi_new - 68)
        cooling_persistence = self.cooling_effects[action] * 0.3  # Cooling effect persists
        cbt_new = cbt_current + thi_effect - cooling_persistence
        cbt_new += self.np_random.normal(0, 0.1)

        # Lying behavior - reduced when stressed
        stress = self._get_stress_level(thi_new, cbt_new)
        lying_base = 0.6 if hour_new < 6 or hour_new > 20 else 0.45
        lying_new = lying_base - 0.05 * stress + self.np_random.normal(0, 0.05)

        # Activity
        activity_new = 0.35 + 0.1 * (1 - stress/3) + self.np_random.normal(0, 0.05)

        # Milk yield - affected by cumulative stress
        stress_impact = 1 - 0.02 * self.cumulative_heat_load
        milk_new = milk * (0.95 + 0.1 * stress_impact) + self.np_random.normal(0, 0.5)

        return np.array([
            np.clip(thi_new, 50, 90),
            np.clip(cbt_new, 38.0, 41.0),
            np.clip(lying_new, 0, 1),
            np.clip(activity_new, 0, 1),
            hour_new,
            np.clip(milk_new, 10, 50),
            self.cumulative_heat_load,
            float(action)
        ])


# Register environment
gym.register(
    id="HeatStressV2-v0",
    entry_point="src.environment.heat_stress_env_v2:HeatStressEnvV2",
)
