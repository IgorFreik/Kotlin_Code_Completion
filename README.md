# Kotlin Code Completion with Phi-1.5

A Python package for fine-tuning language models (specifically Phi-1.5) on code completion tasks for underrepresented programming languages, with a focus on Kotlin.

## Overview

This project explores performance improvements achieved by fine-tuning Large Language Models (LLMs) for code completion in Kotlin, using both a custom Kotlin dataset and Python comparison data from CodeXGLUE.

## Datasets

### 1. Python Dataset
- **Source**: Microsoft's `code_method_completion` dataset
- **Purpose**: Baseline comparison and evaluation
- **Format**: Method completion with docstrings

### 2. Kotlin Dataset
- **Source**: JetBrains/Kotlin GitHub repository
- **Purpose**: Primary target for fine-tuning
- **Processing**: Automatic extraction from `.kt` and `.kts` files

## Evaluation Metrics

The project uses two metrics as suggested by CodeXGLUE:

1. **Exact Match (EM)**: Percentage of predictions that exactly match the ground truth
2. **Edit Similarity**: Levenshtein-based fuzzy string matching score

## Results

### Python (CodeXGLUE)
| Model | Exact Match | Edit Similarity |
|-------|-------------|-----------------|
| Phi-1.5 (pre-trained) | 20.3% | 40.35 |
| Phi-1.5 (fine-tuned) | TBD | TBD |

### Kotlin (Custom Dataset)
| Model | Exact Match | Edit Similarity |
|-------|-------------|-----------------|
| Phi-1.5 (pre-trained) | 9.3% | 45.83 |
| Phi-1.5 (fine-tuned) | TBD | TBD |

## Project Structure

```
├── config.py              # Centralized configuration and constants
├── data_parser.py         # Dataset parsing and preparation
├── trainer.py             # Model training pipeline
├── predictor.py           # Inference and prediction generation
├── evaluator.py           # Evaluation metrics calculation
├── utils.py               # Utility functions for data loading
├── requirements.txt       # Python dependencies
├── data/                  # Dataset storage directory
├── weights/               # Model weights and checkpoints
├── predictions/           # Generated predictions
└── logs/                  # Training logs and TensorBoard data
```

## Usage

### Quick Start with Scripts

```bash
# Parse datasets
python scripts/parse_data.py --type python kotlin

# Train a model
python scripts/train.py --model_name microsoft/phi-1_5 --num_train_epochs 3

# Generate predictions
python scripts/predict.py --model_name microsoft/phi-1_5 --test_size 100

# Evaluate predictions
python scripts/evaluate.py --answers ground_truth.json --predictions predictions.json
```

### Using as an Installed Package

After installation with `pip install -e .`, you can use the command-line tools:

```bash
# Parse datasets
kt-parse --type python kotlin

# Train a model
kt-train --model_name microsoft/phi-1_5 --num_train_epochs 3

# Generate predictions
kt-predict --model_name microsoft/phi-1_5 --test_size 100

# Evaluate predictions
kt-evaluate --answers ground_truth.json --predictions predictions.json
```

### Detailed Usage Examples

#### 1. Dataset Preparation

```bash
# Parse both Python and Kotlin datasets
python scripts/parse_data.py --type python kotlin

# Parse only Kotlin dataset
python scripts/parse_data.py --type kotlin
```

#### 2. Model Training

```bash
# Basic training with default parameters
python scripts/train.py --model_name microsoft/phi-1_5

# Advanced training with custom parameters
python scripts/train.py \
    --model_name microsoft/phi-1_5 \
    --num_train_epochs 10 \
    --train_batch_size 4 \
    --eval_batch_size 4 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 2
```

#### 3. Generate Predictions

```bash
# Using fine-tuned model
python scripts/predict.py \
    --model_name microsoft/phi-1_5 \
    --model_path weights/pytorch_model.bin \
    --test_size 100

# Using base model only
python scripts/predict.py \
    --model_name microsoft/phi-1_5 \
    --test_size 100
```

#### 4. Evaluation

```bash
# Evaluate predictions
python scripts/evaluate.py \
    --answers path/to/ground_truth.json \
    --predictions path/to/predictions.json \
    --prediction_format json \
    --answer_format json
```

## Configuration

All project settings are centralized in `src/kotlin_completion/config.py`. Key configurations include:

- **Dataset paths**: `KT_DS_PATH`, `PY_DS_PATH`
- **Model settings**: `DEFAULT_MODEL_NAME`, `WEIGHTS_DIR`
- **Training hyperparameters**: `DEFAULT_LEARNING_RATE`, `DEFAULT_NUM_EPOCHS`
- **Generation settings**: `DEFAULT_TOKEN_LIMIT`, `DEFAULT_MAX_NEW_TOKENS`

## Development

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_config.py
```

### Package Structure

The project follows Python packaging best practices:

- **`src/` layout**: All source code is in the `src/` directory
- **Proper imports**: Uses relative imports within the package
- **Entry points**: Console scripts for easy command-line usage
- **Tests**: Separate test directory with proper test structure
