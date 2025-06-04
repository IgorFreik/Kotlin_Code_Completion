.PHONY: help install install-dev test clean lint format parse-data train predict evaluate

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package
	pip install -e .

install-dev:  ## Install the package with development dependencies
	pip install -e .[dev]
	pip install -r requirements.txt

test:  ## Run tests
	python -m pytest tests/ -v

clean:  ## Clean up generated files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/

lint:  ## Run linting
	flake8 src/ --max-line-length=100
	mypy src/

format:  ## Format code
	black src/ tests/ scripts/

parse-data:  ## Parse datasets
	python scripts/parse_data.py --type python kotlin

train:  ## Train model with default settings
	python scripts/train.py --model_name microsoft/phi-1_5

predict:  ## Generate predictions
	python scripts/predict.py --model_name microsoft/phi-1_5 --test_size 100

evaluate:  ## Evaluate predictions (requires predictions file)
	python scripts/evaluate.py --answers data/kt_ds.json --predictions predictions/latest.json

setup-dirs:  ## Create necessary directories
	mkdir -p data weights logs predictions 