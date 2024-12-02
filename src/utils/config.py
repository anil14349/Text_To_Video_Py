import yaml
from typing import Dict, Any
import os

class Config:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
            
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def get_model_config(self) -> Dict[str, Any]:
        """Get model-related configuration."""
        return self.config.get('model', {})

    def get_generation_config(self) -> Dict[str, Any]:
        """Get text generation configuration."""
        return self.config.get('generation', {})

    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration."""
        return self.config.get('training', {})

    def get_paths_config(self) -> Dict[str, Any]:
        """Get paths configuration."""
        return self.config.get('paths', {})

    def get_tts_config(self) -> Dict[str, Any]:
        """Get TTS configuration."""
        return self.config.get('tts', {})
        
    def get_prompts_config(self) -> Dict[str, Any]:
        """Get prompts configuration."""
        return self.config.get('prompts', {})
