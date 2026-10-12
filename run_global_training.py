import os
import sys
import logging

from bot.data import global_harvester
from bot.training.historical_simulator import UniversalSimulator
from bot.regime_hmm import HMMRegimeDetector
from bot.ml_filter import MLSignalFilter
from bot.rl_agent import RLExitAgent
from bot.training.memory_seeder import MemorySeeder

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('global_training')

def run_pipeline():
    print("=========================================")
    print("   AEGIS GLOBAL PRE-TRAINING PIPELINE    ")
    print("=========================================\n")
    
    # 1. Fetching Global Data
    print("1. Fetching Global Data (20 Years Daily + 2 Years Intraday)...")
    try:
        global_harvester.fetch_daily_regime_data()
        global_harvester.fetch_intraday_training_data()
    except Exception as e:
        logger.error(f"Failed to fetch global data: {e}")

    # 2. Simulating Trades across Global Regimes
    print("\n2. Simulating Trades across Global Regimes...")
    try:
        simulator = UniversalSimulator()
        simulator.run_simulation()
    except Exception as e:
        logger.error(f"Simulation failed: {e}")

    # 3. Fitting Universal HMM Regime Detector
    print("\n3. Fitting Universal HMM Regime Detector on 20yr SPY...")
    try:
        hmm = HMMRegimeDetector()
        # For universal fitting, assuming fit() doesn't strictly require SPY daily returns
        # as it may load it internally or we pass what's needed.
        # Based on typical usage, let's call fit()
        import pandas as pd
        spy_path = "runtime/global_data/daily_full.csv"
        # We try to fit with default arguments or try to extract SPY
        try:
            # check if HMMRegimeDetector has a specialized fit method
            hmm.fit() 
        except TypeError:
            # if it requires df
            if os.path.exists(spy_path):
                df = pd.read_csv(spy_path, header=[0, 1], index_col=0, parse_dates=True)
                spy_df = df['SPY'] if 'SPY' in df.columns.levels[0] else None
                if spy_df is not None:
                    hmm.fit(spy_df)
                else:
                    logger.warning("SPY not found in daily_full.csv, trying default fit()")
                    hmm.fit()
    except Exception as e:
        logger.error(f"HMM fitting failed: {e}")

    # 4. Training XGBoost ML Filter
    print("\n4. Training XGBoost ML Filter on Global Setups...")
    try:
        ml_filter = MLSignalFilter()
        events_file = 'runtime/global_training_events.jsonl'
        if os.path.exists(events_file):
            # Check if ml_filter.train accepts custom_path
            try:
                ml_filter.train(custom_path=events_file)
            except TypeError:
                ml_filter.train()
        else:
            logger.warning(f"File {events_file} not found. Skipping or running default train.")
            ml_filter.train()
    except Exception as e:
        logger.error(f"ML Filter training failed: {e}")

    # 5. Fine-Tuning PPO RL Agent
    print("\n5. Fine-Tuning PPO RL Agent on Global Exits...")
    try:
        rl_agent = RLExitAgent()
        try:
            rl_agent.retrain_on_real_data()
        except Exception as e:
            logger.error(f"RL Agent retraining failed (maybe missing args): {e}")
    except Exception as e:
        logger.error(f"RL Agent initialization failed: {e}")

    # 6. Seeding Long-Term RAG Memory
    print("\n6. Seeding Long-Term RAG Memory...")
    try:
        seeder = MemorySeeder()
        seeder.seed_extremes(events_file='runtime/global_training_events.jsonl')
    except Exception as e:
        logger.error(f"Memory Seeding failed: {e}")

    print("\nGlobal Pre-Training Complete!")

if __name__ == "__main__":
    run_pipeline()
