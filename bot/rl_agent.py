import os
import json
import glob
from datetime import datetime
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class TradeExitEnv(gym.Env):
    """Synthetic environment to train the RL exit agent."""
    def __init__(self, trade_histories=None):
        super(TradeExitEnv, self).__init__()
        self.action_space = spaces.Discrete(2) # 0: Hold, 1: Close
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)
        self.trade_histories = trade_histories
        self.current_step = 0
        self.pnl = 0.0
        self.current_trajectory = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        if self.trade_histories and len(self.trade_histories) > 0:
            idx = np.random.randint(len(self.trade_histories))
            self.current_trajectory = self.trade_histories[idx]
            self.pnl, atr, bars_held = self.current_trajectory[0]
            # Provide default regime_score=0.0 and garch_vol=0.0
            return np.array([self.pnl, atr, float(bars_held)/50.0, 0.0, 0.0], dtype=np.float32), {}
        else:
            self.current_trajectory = None
            self.pnl = 0.0
            return np.array([self.pnl, 0.02, 0.0, 0.0, 0.0], dtype=np.float32), {}

    def step(self, action):
        done = False
        reward = 0.0
        
        if self.current_trajectory is not None:
            self.current_step += 1
            if self.current_step >= len(self.current_trajectory):
                done = True
                self.current_step = len(self.current_trajectory) - 1
            
            pnl, atr, bars_held = self.current_trajectory[self.current_step]
            
            if action == 1: # Close
                done = True
                reward = pnl * 100
            elif done:
                reward = pnl * 100
            else:
                reward = -0.01
                
            obs = np.array([pnl, atr, float(bars_held)/50.0, 0.0, 0.0], dtype=np.float32)
            return obs, reward, done, False, {}
        else:
            self.current_step += 1
            self.pnl += np.random.normal(0.0005, 0.005) # more realistic random walk
            
            if action == 1: # Close
                done = True
                reward = self.pnl * 100
            elif self.current_step > 50:
                done = True
                reward = self.pnl * 100
            elif self.pnl < -0.02:
                done = True
                reward = -2.0
                
            if not done:
                reward = -0.01

            obs = np.array([self.pnl, 0.02, float(self.current_step)/50.0, 0.0, 0.0], dtype=np.float32)
            return obs, reward, done, False, {}

class RLExitAgent:
    def __init__(self, model_path="runtime/ppo_exit_agent.zip"):
        self.model_path = model_path
        self.model = None
        self.trained = False
        self.load()

    def load(self):
        try:
            from stable_baselines3 import PPO
            if os.path.exists(self.model_path):
                self.model = PPO.load(self.model_path)
                self.trained = True
        except ImportError:
            pass

    def train(self):
        try:
            from stable_baselines3 import PPO
            env = TradeExitEnv()
            self.model = PPO("MlpPolicy", env, verbose=0)
            self.model.learn(total_timesteps=10000)
            self.model.save(self.model_path)
            self.trained = True
        except ImportError:
            pass

    def retrain_on_real_data(self):
        """Load recent trades from runtime/session_events and retrain."""
        try:
            from dateutil.parser import parse as parse_date
            events = []
            for file in glob.glob("runtime/session_events*.jsonl"):
                with open(file, 'r') as f:
                    for line in f:
                        if 'position_opened' in line or 'position_closed' in line:
                            try:
                                events.append(json.loads(line))
                            except json.JSONDecodeError:
                                pass
            
            opened = {}
            histories = []
            for ev in events:
                if ev.get('type') == 'position_opened':
                    opened[ev['payload'].get('entry_order_id')] = ev
                elif ev.get('type') == 'position_closed':
                    open_ev = opened.get(ev['payload'].get('entry_order_id'))
                    if open_ev:
                        try:
                            t1 = parse_date(open_ev['timestamp'])
                            t2 = parse_date(ev['timestamp'])
                            bars_held = max(1, int((t2 - t1).total_seconds() / 60))
                            
                            entry_price = float(open_ev['payload']['entry_price'])
                            pnl_val = float(ev['payload']['pnl'])
                            atr = float(open_ev['payload'].get('atr_at_entry', 0.02))
                            
                            qty = float(open_ev['payload'].get('qty', 1))
                            if qty == 0: qty = 1
                            pct_return = pnl_val / (entry_price * qty)
                            
                            traj = []
                            for i in range(bars_held + 1):
                                current_pct = pct_return * (i / bars_held)
                                traj.append((current_pct, atr, i))
                            
                            if traj:
                                histories.append(traj)
                        except Exception:
                            pass
            
            if histories:
                histories = histories[-50:]
                from stable_baselines3 import PPO
                env = TradeExitEnv(trade_histories=histories)
                if self.model is not None:
                    self.model.set_env(env)
                else:
                    self.model = PPO("MlpPolicy", env, verbose=0)
                self.model.learn(total_timesteps=10000)
                self.model.save(self.model_path)
                self.trained = True
        except Exception as e:
            print(f"Error in retrain_on_real_data: {e}")
            pass

    def should_exit(self, unrealized_pnl_pct: float, current_atr: float, bars_held: int) -> bool:
        if not self.trained or self.model is None:
            return False
        # default regime_score=0.0 and garch_vol=0.0
        obs = np.array([unrealized_pnl_pct, current_atr, float(bars_held)/50.0, 0.0, 0.0], dtype=np.float32)
        action, _ = self.model.predict(obs, deterministic=True)
        return action == 1

if __name__ == "__main__":
    agent = RLExitAgent()
    print("Training DRL Exit Agent...")
    agent.train()
    print("Done!")
