import os
import json
import shutil
from datetime import datetime
from typing import Dict, Any, List, Optional
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class ModelManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.paths_config = config.get_paths_config()
        self.models_dir = os.path.join(self.paths_config.get('output_dir', 'resume_model'), 'versions')
        os.makedirs(self.models_dir, exist_ok=True)

    def save_model_version(self, model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer, 
                         metrics: Dict[str, float], version_name: Optional[str] = None) -> str:
        """Save a version of the model with metrics and metadata."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_name = version_name or f"model_v{timestamp}"
        version_dir = os.path.join(self.models_dir, version_name)
        
        # Save model and tokenizer
        model.save_pretrained(version_dir)
        tokenizer.save_pretrained(version_dir)
        
        # Save metrics and metadata
        metadata = {
            "timestamp": timestamp,
            "metrics": metrics,
            "model_name": model.config.model_type,
            "version": version_name
        }
        
        with open(os.path.join(version_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
            
        print(f"Model version {version_name} saved successfully")
        return version_name

    def load_model_version(self, version_name: str) -> tuple:
        """Load a specific version of the model."""
        version_dir = os.path.join(self.models_dir, version_name)
        
        if not os.path.exists(version_dir):
            raise ValueError(f"Model version {version_name} not found")
            
        model = GPT2LMHeadModel.from_pretrained(version_dir)
        tokenizer = GPT2Tokenizer.from_pretrained(version_dir)
        
        with open(os.path.join(version_dir, "metadata.json"), "r") as f:
            metadata = json.load(f)
            
        return model, tokenizer, metadata

    def list_model_versions(self) -> List[Dict[str, Any]]:
        """List all available model versions with their metadata."""
        versions = []
        
        for version_name in os.listdir(self.models_dir):
            metadata_path = os.path.join(self.models_dir, version_name, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                versions.append(metadata)
                
        return sorted(versions, key=lambda x: x["timestamp"], reverse=True)

    def delete_model_version(self, version_name: str):
        """Delete a specific model version."""
        version_dir = os.path.join(self.models_dir, version_name)
        
        if not os.path.exists(version_dir):
            raise ValueError(f"Model version {version_name} not found")
            
        shutil.rmtree(version_dir)
        print(f"Model version {version_name} deleted successfully")
