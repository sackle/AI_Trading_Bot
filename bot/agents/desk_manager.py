"""
Desk Manager: The central nervous system of the Multi-Agent Firm (Auto-Company model).
It manages the shared memory file (`consensus.md`) that all agents read from and write to.
"""
import os
import re
from pathlib import Path

import config

class DeskManager:
    def __init__(self):
        self.consensus_path = Path(config.ROOT_DIR) / "runtime" / "consensus.md"
        self._ensure_file_exists()
        
    def _ensure_file_exists(self):
        """Creates the initial consensus.md if it doesn't exist."""
        if not self.consensus_path.exists():
            initial_state = (
                "# Multi-Agent Investment Firm - State\n\n"
                "## Market Regime\n"
                "UNKNOWN\n\n"
                "## Active Debates\n"
                "None.\n\n"
                "## Open Positions\n"
                "None.\n\n"
                "## Latest News\n"
                "None.\n"
            )
            self.consensus_path.write_text(initial_state)
            
    def read_consensus(self) -> str:
        """Returns the full text of the consensus file."""
        return self.consensus_path.read_text()
        
    def get_section(self, section_name: str) -> str:
        """Extracts a specific markdown section by its H2 header."""
        content = self.read_consensus()
        # Find the section and capture everything until the next H2 header (or EOF)
        pattern = rf"## {re.escape(section_name)}\n(.*?)(?=\n## |\Z)"
        match = re.search(pattern, content, flags=re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""
        
    def update_section(self, section_name: str, new_content: str):
        """Replaces the content of a specific markdown section."""
        content = self.read_consensus()
        pattern = rf"(## {re.escape(section_name)}\n)(.*?)(?=\n## |\Z)"
        
        if re.search(pattern, content, flags=re.DOTALL):
            # Replace existing section
            new_text = re.sub(
                pattern, 
                rf"\1{new_content}\n", 
                content, 
                flags=re.DOTALL, 
                count=1
            )
            self.consensus_path.write_text(new_text)
        else:
            # Append new section if it doesn't exist
            with open(self.consensus_path, "a") as f:
                f.write(f"\n## {section_name}\n{new_content}\n")
