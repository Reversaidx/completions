from pydantic import BaseModel
import yaml
from pathlib import Path


class ModelConfig(BaseModel):
    hidden_dim: int = 128
    dropout: float = 0.5
    batch_size: int = 64
    learning_rate: float = 0.002
    max_length: int = 512
    num_epochs: int = 5


def load_config(config_path: str = "configs/model.yaml") -> ModelConfig:
    """Загружает конфигурацию из YAML файла"""
    config_file = Path(config_path)
    
    if not config_file.exists():
        print(f"Config file {config_path} not found, using defaults")
        return ModelConfig()
    
    with open(config_file, 'r') as f:
        config_data = yaml.safe_load(f)
    
    return ModelConfig(**config_data['model'])