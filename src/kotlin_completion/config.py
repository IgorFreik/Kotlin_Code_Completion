"""
Configuration file for the Kotlin Code Completion project.
This file contains all constants and configuration settings used across the project.
"""

import os

# Data paths and constants
DATA_FOLDER = 'data'
PY_DS_PATH = os.path.join(DATA_FOLDER, 'py_ds.json')
KT_DS_PATH = os.path.join(DATA_FOLDER, 'kt_ds.json')
KT_FILES_DIR = os.path.join(DATA_FOLDER, 'kotlin-master')

# GitHub repository settings
KT_REPO_NAME = 'kotlin'
KT_REPO_OWNER = 'jetbrains'

# Model and training settings
WEIGHTS_DIR = 'weights/'
DEFAULT_MODEL_NAME = 'microsoft/phi-1_5'

# File extensions
KOTLIN_EXTENSIONS = {'.kt', '.kts'}

# Training hyperparameters (defaults)
DEFAULT_TRAIN_BATCH_SIZE = 2
DEFAULT_EVAL_BATCH_SIZE = 2
DEFAULT_LEARNING_RATE = 1e-5
DEFAULT_NUM_EPOCHS = 3
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 2

# Prediction settings
DEFAULT_TOKEN_LIMIT = 1000
DEFAULT_MAX_NEW_TOKENS = 50
DEFAULT_TEMPERATURE = 0.7

# Dataset settings
DATASET_SHARD_SIZE = 40
DATASET_SHARD_INDEX = 0
TEST_SIZE_RATIO = 0.3
EVAL_TEST_SPLIT_RATIO = 0.5

# Random seed for reproducibility
RANDOM_SEED = 42

# Logging configuration
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Output directories
PREDICTIONS_DIR = 'predictions'
LOGS_DIR = 'logs' 