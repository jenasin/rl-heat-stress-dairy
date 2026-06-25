"""
Heat Stress Management Environment for Dairy Cattle
====================================================
Custom Gymnasium environment for training RL agents to make
optimal cooling decisions based on multimodal sensor data.

Paper: Deep Reinforcement Learning Agent for Autonomous Heat Stress
       Intervention in Dairy Cattle: Learning Optimal Cooling Strategies
       from Multimodal Sensor Data

Authors:
    Jan Saro, Ing., Ph.D., MBA - Czech University of Life Sciences Prague
    Jana Mazancová, Ph.D. - Czech University of Life Sciences Prague
    Jaromír Ducháček, Ing., Ph.D. - Czech University of Life Sciences Prague

Dataset: MmCows (NeurIPS 2024)
Journal: Livestock Science
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any


class HeatStressEnv(gym.Env):
    """
    Gymnasium environment for heat stress management in dairy cattle.
    
    State Space (6 features):
        - THI: Temperature-Humidity Index [50, 90]
        - CBT: Core Body Temperature [38.0, 41.0]
        - lying_ratio: Proportion of time lying [0, 1]
        - activity: Movement intensity [0, 1]
        - hour: Hour of day [0, 23]
        - milk_yield_prev: Previous milk yield [0, 50]
    
    Action Space (4 discrete actions):
        0: No intervention
        1: Mild cooling (fans)
        2: Intensive cooling (fans + sprinklers)
        3: Water spray only
    
    Reward:
        R = α * Δmilk_yield + β * comfort_score - γ * energy_cost
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(
        self,
        data: Optional[pd.DataFrame] = None,
        reward_weights: Dict[str, float] = None,
        episode_length: int = 48,  # 48 hours = 2 days
        render_mode: Optional[str] = None
    ):
        super().__init__()
        
        self.data = data
        self.episode_length = episode_length
        self.render_mode = render_mode
        
        # Default reward weights
        self.reward_weights = reward_weights or {
            "milk": 1.0,
            "comfort": 0.5,
            "energy": 0.3
        }
        
        # State space: [THI, CBT, lying_ratio, activity, hour, milk_prev]
        self.observation_space = spaces.Box(
            low=np.array([50.0, 38.0, 0.0, 0.0, 0.0, 0.0]),
            high=np.array([90.0, 41.0, 1.0, 1.0, 23.0, 50.0]),
            dtype=np.float32
        )
        
        # Action space: 4 discrete cooling interventions
        self.action_space = spaces.Discrete(4)
        
        # Action energy costs (relative units)
        self.energy_costs = {
            0: 0.0,    # No action
            1: 0.3,    # Fans
            2: 1.0,    # Fans + sprinklers
            3: 0.5     # Water spray
        }
        
        # Action cooling effects (temperature reduction in °C)
        self.cooling_effects = {
            0: 0.0,
            1: 0.3,
            2: 0.8,
            3: 0.4
        }
        
        # Episode tracking
        self.current_step = 0
        self.current_state = None
        self.cow_id = None
        self.episode_data = None
        
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        self.current_step = 0
        
        if self.data is not None:
            # Sample a random cow and time period from real data
            cow_ids = self.data['cow_id'].unique()
            self.cow_id = self.np_random.choice(cow_ids)
            cow_data = self.data[self.data['cow_id'] == self.cow_id]
            
            # Random start point
            max_start = len(cow_data) - self.episode_length
            if max_start > 0:
                start_idx = self.np_random.integers(0, max_start)
                self.episode_data = cow_data.iloc[start_idx:start_idx + self.episode_length]
            else:
                self.episode_data = cow_data
            
            self.current_state = self._get_state_from_data(0)
        else:
            # Synthetic initial state for testing
            self.current_state = self._generate_synthetic_state()
        
        info = {"cow_id": self.cow_id, "step": 0}
        return self.current_state.astype(np.float32), info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one time step."""
        assert self.action_space.contains(action), f"Invalid action: {action}"
        
        # Get current state components
        thi, cbt, lying, activity, hour, milk_prev = self.current_state
        
        # Apply cooling effect to CBT
        cbt_new = max(38.0, cbt - self.cooling_effects[action])
        
        # Calculate comfort score (0-1, higher is better)
        comfort = self._calculate_comfort(thi, cbt_new, lying)
        
        # Calculate milk yield change based on comfort
        milk_change = self._calculate_milk_change(comfort, cbt, cbt_new)
        
        # Calculate reward
        reward = (
            self.reward_weights["milk"] * milk_change +
            self.reward_weights["comfort"] * comfort -
            self.reward_weights["energy"] * self.energy_costs[action]
        )
        
        # Move to next step
        self.current_step += 1
        
        # Get next state
        if self.data is not None and self.current_step < len(self.episode_data):
            self.current_state = self._get_state_from_data(self.current_step)
            # Update CBT based on action taken
            self.current_state[1] = cbt_new
        else:
            self.current_state = self._transition_state(action)
        
        # Check termination
        terminated = self.current_step >= self.episode_length
        truncated = False
        
        info = {
            "cow_id": self.cow_id,
            "step": self.current_step,
            "thi": thi,
            "cbt": cbt_new,
            "comfort": comfort,
            "milk_change": milk_change,
            "energy_cost": self.energy_costs[action],
            "action_name": self._action_name(action)
        }
        
        return self.current_state.astype(np.float32), reward, terminated, truncated, info
    
    def _get_state_from_data(self, idx: int) -> np.ndarray:
        """Extract state from real data."""
        row = self.episode_data.iloc[idx]
        return np.array([
            row.get('thi', 70.0),
            row.get('cbt', 38.5),
            row.get('lying_ratio', 0.5),
            row.get('activity', 0.3),
            row.get('hour', 12),
            row.get('milk_yield', 30.0)
        ])
    
    def _generate_synthetic_state(self) -> np.ndarray:
        """Generate synthetic state for testing."""
        hour = self.np_random.integers(0, 24)
        # THI follows daily pattern (higher during day)
        thi_base = 65 + 15 * np.sin(np.pi * (hour - 6) / 12) if 6 <= hour <= 18 else 60
        thi = thi_base + self.np_random.normal(0, 3)
        thi = np.clip(thi, 50, 90)
        
        # CBT correlates with THI
        cbt = 38.5 + 0.03 * (thi - 68) + self.np_random.normal(0, 0.2)
        cbt = np.clip(cbt, 38.0, 41.0)
        
        # Lying ratio inversely related to heat stress
        lying = 0.6 - 0.01 * max(0, thi - 68) + self.np_random.normal(0, 0.1)
        lying = np.clip(lying, 0, 1)
        
        # Activity
        activity = 0.4 + self.np_random.normal(0, 0.1)
        activity = np.clip(activity, 0, 1)
        
        # Previous milk yield
        milk_prev = 30 - 0.5 * max(0, thi - 72) + self.np_random.normal(0, 2)
        milk_prev = np.clip(milk_prev, 0, 50)
        
        return np.array([thi, cbt, lying, activity, hour, milk_prev])
    
    def _transition_state(self, action: int) -> np.ndarray:
        """Transition to next state (synthetic)."""
        thi, cbt, lying, activity, hour, milk_prev = self.current_state
        
        # Next hour
        hour_new = (hour + 1) % 24
        
        # THI changes with time of day
        thi_new = 65 + 15 * np.sin(np.pi * (hour_new - 6) / 12) if 6 <= hour_new <= 18 else 60
        thi_new += self.np_random.normal(0, 2)
        thi_new = np.clip(thi_new, 50, 90)
        
        # CBT affected by THI and previous cooling
        cbt_new = cbt + 0.02 * (thi_new - 68) - self.cooling_effects[action] * 0.5
        cbt_new += self.np_random.normal(0, 0.1)
        cbt_new = np.clip(cbt_new, 38.0, 41.0)
        
        # Lying ratio
        lying_new = 0.6 - 0.01 * max(0, thi_new - 68) + self.np_random.normal(0, 0.05)
        lying_new = np.clip(lying_new, 0, 1)
        
        # Activity
        activity_new = activity + self.np_random.normal(0, 0.05)
        activity_new = np.clip(activity_new, 0, 1)
        
        # Milk yield based on comfort
        comfort = self._calculate_comfort(thi_new, cbt_new, lying_new)
        milk_new = milk_prev * (0.9 + 0.2 * comfort) + self.np_random.normal(0, 1)
        milk_new = np.clip(milk_new, 0, 50)
        
        return np.array([thi_new, cbt_new, lying_new, activity_new, hour_new, milk_new])
    
    def _calculate_comfort(self, thi: float, cbt: float, lying: float) -> float:
        """Calculate comfort score (0-1)."""
        # THI component (optimal < 68)
        thi_score = 1.0 - min(1.0, max(0, thi - 68) / 22)
        
        # CBT component (optimal 38.5)
        cbt_score = 1.0 - min(1.0, abs(cbt - 38.5) / 2.5)
        
        # Lying component (normal ~0.5-0.6)
        lying_score = 1.0 - abs(lying - 0.55) * 2
        lying_score = max(0, min(1, lying_score))
        
        # Weighted average
        comfort = 0.4 * thi_score + 0.4 * cbt_score + 0.2 * lying_score
        return comfort
    
    def _calculate_milk_change(self, comfort: float, cbt_old: float, cbt_new: float) -> float:
        """Calculate change in milk yield potential."""
        # Base effect from comfort
        base_change = (comfort - 0.7) * 2  # Centered around 0.7 comfort
        
        # Bonus for reducing CBT during heat stress
        cbt_improvement = (cbt_old - cbt_new) if cbt_old > 39.0 else 0
        
        return base_change + cbt_improvement * 0.5
    
    def _action_name(self, action: int) -> str:
        """Get human-readable action name."""
        names = {
            0: "no_action",
            1: "mild_cooling",
            2: "intensive_cooling",
            3: "water_spray"
        }
        return names.get(action, "unknown")
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            thi, cbt, lying, activity, hour, milk = self.current_state
            print(f"\nStep {self.current_step}:")
            print(f"  THI: {thi:.1f} | CBT: {cbt:.2f}°C | Lying: {lying:.2%}")
            print(f"  Hour: {int(hour):02d}:00 | Milk: {milk:.1f} kg")
            
            # Heat stress indicator
            if thi >= 79:
                stress = "🔴 SEVERE"
            elif thi >= 72:
                stress = "🟠 MODERATE"
            elif thi >= 68:
                stress = "🟡 MILD"
            else:
                stress = "🟢 NONE"
            print(f"  Heat Stress: {stress}")


# Register the environment
gym.register(
    id="HeatStressManagement-v0",
    entry_point="src.environment.heat_stress_env:HeatStressEnv",
)


if __name__ == "__main__":
    # Test the environment
    env = HeatStressEnv(render_mode="human")
    obs, info = env.reset(seed=42)
    
    print("Testing Heat Stress Environment")
    print("=" * 50)
    
    total_reward = 0
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        env.render()
        print(f"  Action: {info['action_name']} | Reward: {reward:.3f}")
        
        if terminated or truncated:
            break
    
    print(f"\nTotal reward: {total_reward:.3f}")
