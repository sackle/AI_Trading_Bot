import sqlite3
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger('aegis.memory')

class TradingMemory:
    def __init__(self, db_path: str = 'runtime/aegis_memory.db'):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()
    
    def _create_tables(self):
        """Create all memory tables if they don't exist."""
        self.conn.executescript('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                strategy TEXT NOT NULL,
                entry_price REAL,
                exit_price REAL,
                realized_pnl REAL,
                position_size REAL,
                regime TEXT,
                garch_vol REAL,
                committee_approved INTEGER,
                virtual_brain_approved INTEGER,
                ml_win_prob REAL,
                holding_bars INTEGER,
                exit_reason TEXT,
                metadata TEXT  -- JSON blob for extra context
            );
            
            CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                decision_type TEXT NOT NULL,  -- 'virtual_brain', 'committee', 'guardian', 'ml_filter'
                approved INTEGER NOT NULL,
                reasoning TEXT,
                context TEXT  -- JSON blob
            );
            
            CREATE TABLE IF NOT EXISTS regime_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                regime TEXT NOT NULL,
                confidence REAL,
                trending_prob REAL,
                mean_reverting_prob REAL,
                crisis_prob REAL
            );
            
            CREATE TABLE IF NOT EXISTS playbook_changes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                action TEXT NOT NULL,  -- 'added', 'removed', 'promoted'
                rule_name TEXT NOT NULL,
                rule_content TEXT,
                source TEXT  -- 'human', 'post_mortem', 'weekend_analyzer'
            );
            
            CREATE TABLE IF NOT EXISTS model_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                model_name TEXT NOT NULL,  -- 'xgboost', 'ppo_exit', 'hmm'
                version TEXT,
                trigger TEXT,  -- 'drift', 'scheduled', 'manual'
                metrics TEXT  -- JSON blob with performance metrics
            );
            
            CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
            CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy);
            CREATE INDEX IF NOT EXISTS idx_trades_regime ON trades(regime);
            CREATE INDEX IF NOT EXISTS idx_decisions_symbol ON decisions(symbol);
            CREATE INDEX IF NOT EXISTS idx_regime_timestamp ON regime_history(timestamp);
        ''')
        self.conn.commit()
    
    # --- Write Methods ---
    
    def record_trade(self, symbol, side, strategy, entry_price, exit_price,
                     realized_pnl, position_size=0, regime='', garch_vol=0.0,
                     committee_approved=None, virtual_brain_approved=None,
                     ml_win_prob=0.0, holding_bars=0, exit_reason='',
                     metadata=None):
        self.conn.execute(
            'INSERT INTO trades (timestamp, symbol, side, strategy, entry_price, exit_price, '
            'realized_pnl, position_size, regime, garch_vol, committee_approved, '
            'virtual_brain_approved, ml_win_prob, holding_bars, exit_reason, metadata) '
            'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
            (datetime.now(timezone.utc).isoformat(), symbol, side, strategy,
             entry_price, exit_price, realized_pnl, position_size, regime,
             garch_vol, committee_approved, virtual_brain_approved, ml_win_prob,
             holding_bars, exit_reason, json.dumps(metadata or {}))
        )
        self.conn.commit()
    
    def record_decision(self, symbol, decision_type, approved, reasoning='', context=None):
        self.conn.execute(
            'INSERT INTO decisions (timestamp, symbol, decision_type, approved, reasoning, context) '
            'VALUES (?, ?, ?, ?, ?, ?)',
            (datetime.now(timezone.utc).isoformat(), symbol, decision_type,
             int(approved), reasoning, json.dumps(context or {}))
        )
        self.conn.commit()
    
    def record_regime(self, regime, confidence, probs):
        self.conn.execute(
            'INSERT INTO regime_history (timestamp, regime, confidence, '
            'trending_prob, mean_reverting_prob, crisis_prob) VALUES (?, ?, ?, ?, ?, ?)',
            (datetime.now(timezone.utc).isoformat(), regime, confidence,
             probs.get('TRENDING', 0), probs.get('MEAN_REVERTING', 0), probs.get('CRISIS', 0))
        )
        self.conn.commit()
    
    def record_playbook_change(self, action, rule_name, rule_content='', source=''):
        self.conn.execute(
            'INSERT INTO playbook_changes (timestamp, action, rule_name, rule_content, source) '
            'VALUES (?, ?, ?, ?, ?)',
            (datetime.now(timezone.utc).isoformat(), action, rule_name, rule_content, source)
        )
        self.conn.commit()
    
    def record_model_version(self, model_name, version, trigger, metrics=None):
        self.conn.execute(
            'INSERT INTO model_versions (timestamp, model_name, version, trigger, metrics) '
            'VALUES (?, ?, ?, ?, ?)',
            (datetime.now(timezone.utc).isoformat(), model_name, version, trigger,
             json.dumps(metrics or {}))
        )
        self.conn.commit()
    
    # --- Query Methods (RAG-style) ---
    
    def query_similar_trades(self, symbol: str, regime: str = '', 
                             strategy: str = '', limit: int = 10) -> list[dict]:
        """Find historical trades matching the current context."""
        query = 'SELECT * FROM trades WHERE symbol = ?'
        params = [symbol]
        if regime:
            query += ' AND regime = ?'
            params.append(regime)
        if strategy:
            query += ' AND strategy = ?'
            params.append(strategy)
        query += ' ORDER BY timestamp DESC LIMIT ?'
        params.append(limit)
        rows = self.conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]
    
    def query_strategy_performance(self, strategy: str, days: int = 30) -> dict:
        """Get performance stats for a strategy over the last N days."""
        rows = self.conn.execute(
            'SELECT realized_pnl FROM trades WHERE strategy = ? '
            'AND timestamp > datetime("now", ?)',
            (strategy, f'-{days} days')
        ).fetchall()
        if not rows:
            return {'trades': 0, 'win_rate': 0, 'total_pnl': 0, 'avg_pnl': 0}
        pnls = [r['realized_pnl'] for r in rows]
        wins = sum(1 for p in pnls if p > 0)
        return {
            'trades': len(pnls),
            'win_rate': wins / len(pnls) if pnls else 0,
            'total_pnl': sum(pnls),
            'avg_pnl': sum(pnls) / len(pnls) if pnls else 0
        }
    
    def query_regime_performance(self, regime: str, limit: int = 50) -> dict:
        """How does the bot perform in a specific regime?"""
        rows = self.conn.execute(
            'SELECT strategy, realized_pnl FROM trades WHERE regime = ? '
            'ORDER BY timestamp DESC LIMIT ?',
            (regime, limit)
        ).fetchall()
        if not rows:
            return {'trades': 0}
        pnls = [r['realized_pnl'] for r in rows]
        wins = sum(1 for p in pnls if p > 0)
        return {
            'trades': len(pnls),
            'win_rate': wins / len(pnls),
            'total_pnl': sum(pnls),
            'best_strategy': max(set(r['strategy'] for r in rows), 
                               key=lambda s: sum(r['realized_pnl'] for r in rows if r['strategy'] == s))
        }
    
    def generate_context_for_llm(self, symbol: str, regime: str = '',
                                  strategy: str = '') -> str:
        """Generate a natural language context block for LLM prompts.
        This is the RAG retrieval function."""
        similar = self.query_similar_trades(symbol, regime, strategy, limit=5)
        if not similar:
            return f'No historical trades found for {symbol} in this context.'
        
        lines = [f'Historical Memory for {symbol} ({regime} regime, {strategy} strategy):']
        for t in similar:
            result = 'WIN' if t['realized_pnl'] > 0 else 'LOSS'
            lines.append(
                f'  - {t["timestamp"][:10]}: {result} ${t["realized_pnl"]:.2f} '
                f'(entry {t["entry_price"]:.2f}, exit {t["exit_price"]:.2f}, '
                f'exit_reason: {t["exit_reason"]})')
        
        perf = self.query_strategy_performance(strategy)
        if perf['trades'] > 0:
            lines.append(
                f'  Strategy "{strategy}" 30-day stats: '
                f'{perf["trades"]} trades, {perf["win_rate"]*100:.0f}% win rate, '
                f'${perf["total_pnl"]:.2f} total PnL')
        
        return '\n'.join(lines)
    
    def get_stats(self) -> dict:
        """Get overall memory stats for dashboard display."""
        total_trades = self.conn.execute('SELECT COUNT(*) FROM trades').fetchone()[0]
        total_decisions = self.conn.execute('SELECT COUNT(*) FROM decisions').fetchone()[0]
        total_regimes = self.conn.execute('SELECT COUNT(*) FROM regime_history').fetchone()[0]
        total_rules = self.conn.execute('SELECT COUNT(*) FROM playbook_changes').fetchone()[0]
        return {
            'total_trades': total_trades,
            'total_decisions': total_decisions,
            'regime_entries': total_regimes,
            'playbook_changes': total_rules
        }
