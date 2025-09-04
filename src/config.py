from pydantic_settings import BaseSettings
import yaml
from pathlib import Path
import torch
torch.cuda.manual_seed_all(42)

class ModelConfig(BaseSettings):
    hidden_dim: int = 128
    dropout: float = 0.5
    batch_size: int = 64
    learning_rate: float = 0.002
    max_length: int = 512
    num_epochs: int = 1

    class Config:
        env_prefix = "MODEL_"
        case_sensitive = False


def get_device() -> str:
    """Определяет лучшее доступное устройство"""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def load_config(config_path: str = "configs/model.yaml") -> ModelConfig:
    """Загружает конфигурацию из YAML файла с поддержкой env переменных"""
    config_file = Path(config_path)
    
    if config_file.exists():
        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)
        return ModelConfig(**config_data['model'])
    else:
        return ModelConfig()