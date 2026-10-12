import sqlite3
import json
import os
from pathlib import Path

def get_command_center_data():
    db_path = "runtime/aegis_memory.db"
    data = {
        "memory_stats": {},
        "current_regime": {},
        "recent_decisions": [],
        "shadow_trader": {},
        "guardian": {"status": "ACTIVE", "consecutive_losses": 0}
    }
    
    if os.path.exists(db_path):
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            
            # Memory Stats
            stats = {}
            for table in ['trades', 'decisions', 'regime_history', 'playbook_changes']:
                try:
                    stats[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                except:
                    stats[table] = 0
            data["memory_stats"] = stats
            
            # Current Regime
            try:
                regime = conn.execute("SELECT * FROM regime_history ORDER BY timestamp DESC LIMIT 1").fetchone()
                if regime:
                    data["current_regime"] = dict(regime)
            except:
                pass
                
            # Recent Decisions
            try:
                decisions = conn.execute("SELECT * FROM decisions ORDER BY timestamp DESC LIMIT 10").fetchall()
                data["recent_decisions"] = [dict(d) for d in decisions]
            except:
                pass
                
            conn.close()
        except Exception as e:
            print(f"Error reading DB: {e}")
            
    # Shadow Trader
    shadow_path = "runtime/shadow_results.json"
    if os.path.exists(shadow_path):
        try:
            with open(shadow_path, "r") as f:
                data["shadow_trader"] = json.load(f)
        except:
            pass
            
    # Guardian State
    guardian_path = "runtime/guardian_emergency.json"
    if os.path.exists(guardian_path):
        try:
            with open(guardian_path, "r") as f:
                data["guardian"] = json.load(f)
        except:
            pass
            
    return data
