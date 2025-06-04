# Kotlin Code Completion with Phi-1.5

A Python project for fine-tuning language models (specifically Phi-1.5) on code completion tasks for underrepresented programming languages, with a focus on Kotlin.

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

### 1. Dataset Preparation

Parse datasets for training and evaluation:

```bash
# Parse both Python and Kotlin datasets
python data_parser.py --type python kotlin

# Parse only Kotlin dataset
python data_parser.py --type kotlin

# Parse only Python dataset
python data_parser.py --type python
```

The datasets follow this format:
```json
{
  "input": "function header() {\n    val x = 10\n    val y =",
  "labels": " 20"
}
```

### 2. Model Training

Fine-tune a model using the prepared datasets:

```bash
# Basic training with default parameters
python trainer.py --model_name microsoft/phi-1_5

# Advanced training with custom parameters
python trainer.py \
    --model_name microsoft/phi-1_5 \
    --num_train_epochs 10 \
    --train_batch_size 4 \
    --eval_batch_size 4 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 2
```

### 3. Generate Predictions

Generate predictions using a trained model:

```bash
# Using fine-tuned model
python predictor.py \
    --model_name microsoft/phi-1_5 \
    --model_path weights/pytorch_model.bin \
    --test_size 100

# Using base model only
python predictor.py \
    --model_name microsoft/phi-1_5 \
    --test_size 100
```

### 4. Evaluation

Evaluate predictions against ground truth:

```bash
# Evaluate predictions
python evaluator.py \
    --answers path/to/ground_truth.json \
    --predictions path/to/predictions.json \
    --prediction_format json \
    --answer_format json
```

## Configuration

All project settings are centralized in `config.py`. Key configurations include:

- **Dataset paths**: `KT_DS_PATH`, `PY_DS_PATH`
- **Model settings**: `DEFAULT_MODEL_NAME`, `WEIGHTS_DIR`
- **Training hyperparameters**: `DEFAULT_LEARNING_RATE`, `DEFAULT_NUM_EPOCHS`
- **Generation settings**: `DEFAULT_TOKEN_LIMIT`, `DEFAULT_MAX_NEW_TOKENS`

