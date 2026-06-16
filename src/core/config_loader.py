import yaml
from pathlib import Path
from typing import Any, Dict

class ConfigLoader:
    """Handles loading and validation of project configuration."""
    
    def __init__(self, config_path: str = "config.yaml"):
        # Resolve absolute path relative to project root
        self.root_dir = Path(__file__).parent.parent.parent
        self.config_path = self.root_dir / config_path
        self._config = self._load_yaml()

    def _load_yaml(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found at {self.config_path}")
        
        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)

    def get(self, key_path: str, default: Any = None) -> Any:
        """Retrieves a value from the config using dot notation (e.g., 'data.raw_pdf_dir')."""
        keys = key_path.split(".")
        value = self._config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return default
            if value is None:
                return default
        return value

    @property
    def all(self) -> Dict[str, Any]:
        return self._config

    def get_path(self, key_path: str) -> Path:
        """Retrieves a path from the config and ensures it's absolute."""
        path_str = self.get(key_path)
        if path_str is None:
            raise KeyError(f"Config key '{key_path}' not found.")
        return self.root_dir / path_str
