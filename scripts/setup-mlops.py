#!/usr/bin/env python3
"""Setup script for ML infrastructure"""

import os
import yaml
import logging
from pathlib import Path

def setup_mlflow():
    """Configure MLflow tracking server"""
    mlflow_config = {
        'artifact_root': './mlruns',
        'tracking_uri': os.getenv('MLFLOW_TRACKING_URI'),
        'registry_uri': 'sqlite:///mlflow.db',
        'experiment_defaults': {
            'artifact_location': './mlruns',
            'lifecycle_stage': 'active'
        }
    }
    
    with open('config/mlflow-config.yml', 'w') as f:
        yaml.dump(mlflow_config, f)

def setup_model_registry():
    """Configure model registry and versioning"""
    registry_config = {
        'storage': {
            'local_storage': './models',
            's3_storage': os.getenv('MODEL_STORAGE_S3', 'none')
        },
        'versioning': {
            'strategy': 'semantic',
            'auto_increment': True
        },
        'validation': {
            'required': True,
            'metrics': ['accuracy', 'loss', 'f1']
        }
    }
    
    with open('config/model-registry.yml', 'w') as f:
        yaml.dump(registry_config, f)

def setup_experiment_tracking():
    """Configure experiment tracking"""
    tracking_config = {
        'wandb': {
            'project': os.getenv('WANDB_PROJECT'),
            'entity': os.getenv('WANDB_ENTITY'),
            'api_key': os.getenv('WANDB_API_KEY'),
            'log_artifacts': True
        },
        'metrics': {
            'training': ['loss', 'accuracy', 'val_loss', 'val_accuracy'],
            'system': ['gpu_usage', 'memory_usage', 'training_time']
        }
    }
    
    with open('config/experiment-tracking.yml', 'w') as f:
        yaml.dump(tracking_config, f)

def main():
    """Main setup function"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create directories
    Path('models').mkdir(exist_ok=True)
    Path('mlruns').mkdir(exist_ok=True)
    Path('config').mkdir(exist_ok=True)
    
    # Setup components
    setup_mlflow()
    setup_model_registry()
    setup_experiment_tracking()
    
    logging.info("ML infrastructure setup completed successfully")

if __name__ == "__main__":
    main()

# In scripts/setup-mlops.py
# This sets up your ML tracking
import mlflow

# Initialize MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.create_experiment("my-experiment")

# Track experiments
with mlflow.start_run():
    # Your ML code here
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_metric("accuracy", 0.95)
