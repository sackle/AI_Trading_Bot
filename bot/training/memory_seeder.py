import json
import logging
from bot.memory import TradingMemory

logger = logging.getLogger('aegis.memory_seeder')

class MemorySeeder:
    def __init__(self):
        self.memory = TradingMemory()
    
    def seed_extremes(self, events_file='runtime/global_training_events.jsonl'):
        logger.info(f"Seeding extreme market events from {events_file} into RAG Memory...")
        try:
            trades = []
            with open(events_file, 'r') as f:
                for line in f:
                    if line.strip():
                        trade_data = json.loads(line)
                        if 'unrealized_plpc' in trade_data or 'realized_pnl' in trade_data:
                            trades.append(trade_data)
            
            if not trades:
                logger.warning("No trades found to seed.")
                return

            # Determine the key to sort by, unrealized_plpc is preferred but realized_pnl is fallback
            sort_key = lambda x: x.get('unrealized_plpc', x.get('realized_pnl', 0.0))
            trades.sort(key=sort_key)
            
            mega_losers = trades[:500]
            mega_winners = trades[-500:] if len(trades) > 500 else []
            
            logger.info(f"Injecting {len(mega_winners)} mega-winners and {len(mega_losers)} mega-losers.")
            
            for trade in mega_losers:
                symbol = trade.get('symbol', 'UNKNOWN')
                self.memory.record_trade(
                    symbol=symbol,
                    side=trade.get('side', 'long'),
                    strategy=trade.get('strategy', 'global_seeder'),
                    entry_price=trade.get('entry_price', 0.0),
                    exit_price=trade.get('exit_price', 0.0),
                    realized_pnl=trade.get('realized_pnl', trade.get('unrealized_plpc', -1.0) * 100),
                    position_size=trade.get('position_size', 0),
                    regime='CRISIS',
                    exit_reason='Seeded Mega-Loser',
                    metadata=trade
                )
            
            for trade in mega_winners:
                symbol = trade.get('symbol', 'UNKNOWN')
                self.memory.record_trade(
                    symbol=symbol,
                    side=trade.get('side', 'long'),
                    strategy=trade.get('strategy', 'global_seeder'),
                    entry_price=trade.get('entry_price', 0.0),
                    exit_price=trade.get('exit_price', 0.0),
                    realized_pnl=trade.get('realized_pnl', trade.get('unrealized_plpc', 1.0) * 100),
                    position_size=trade.get('position_size', 0),
                    regime='TRENDING',
                    exit_reason='Seeded Mega-Winner',
                    metadata=trade
                )
                
            logger.info("Memory seeding complete.")
            print("Memory seeding complete.")
            
        except FileNotFoundError:
            logger.error(f"Events file {events_file} not found. Skipping seeding.")
            print(f"Events file {events_file} not found. Skipping seeding.")
        except Exception as e:
            logger.error(f"Error during memory seeding: {e}")
            print(f"Error during memory seeding: {e}")
