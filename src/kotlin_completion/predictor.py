import argparse
import json
import logging
from datetime import datetime
from typing import Optional

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from . import config
from .evaluator import post_process
from .utils import load_kt_dataset

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL), format=config.LOG_FORMAT)


def single_predict(
    model: AutoModelForCausalLM, 
    tokenizer: AutoTokenizer, 
    input_text: str, 
    token_limit: int = None,
    max_new_tokens: int = None,
    temperature: float = None
) -> str:
    """
    Make a single prediction given model, tokenizer, and input text.
    
    Args:
        model: The model to make predictions with
        tokenizer: The tokenizer for the model
        input_text: The input string with code prefix
        token_limit: The maximum number of input tokens
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Sampling temperature for generation
        
    Returns:
        The predicted string (code line)
    """
    # Use default values from config if not provided
    token_limit = token_limit or config.DEFAULT_TOKEN_LIMIT
    max_new_tokens = max_new_tokens or config.DEFAULT_MAX_NEW_TOKENS
    temperature = temperature or config.DEFAULT_TEMPERATURE
    
    try:
        # Preprocess and tokenize input
        processed_input = post_process(input_text)
        tokens = tokenizer(
            processed_input, 
            return_tensors="pt", 
            return_attention_mask=True,
            truncation=True
        )
        
        input_len = len(tokens['input_ids'][0])
        
        # Truncate if input is too long
        if input_len > token_limit:
            tokens['input_ids'] = tokens['input_ids'][0][-token_limit:].reshape(1, -1)
            tokens['attention_mask'] = tokens['attention_mask'][0][-token_limit:].reshape(1, -1)
            input_len = token_limit
        
        # Generate prediction
        with torch.no_grad():
            outputs = model.generate(
                **tokens,
                max_length=input_len + max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.eos_token_id,
                num_return_sequences=1
            )
        
        # Decode and extract the new tokens
        pred = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        
        # Return only the first line of the prediction
        pred_lines = pred.split('\n')
        return pred_lines[0].strip() if pred_lines else ""
        
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return ""


def load_model_and_tokenizer(model_name: str, model_path: Optional[str] = None) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load model and tokenizer, optionally from a custom checkpoint.
    
    Args:
        model_name: Name of the base model
        model_path: Optional path to custom model weights
        
    Returns:
        Tuple of (model, tokenizer)
    """
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load custom weights if provided
        if model_path:
            logger.info(f"Loading custom weights from {model_path}")
            state_dict = torch.load(model_path, map_location='cpu')
            model.load_state_dict(state_dict)
        
        model.eval()
        
        # Add padding token if it doesn't exist
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        return model, tokenizer
        
    except Exception as e:
        raise RuntimeError(f"Failed to load model {model_name}: {e}")


def save_predictions(predictions: list, model_name: str, output_dir: str = None) -> str:
    """
    Save predictions to a JSON file with timestamp.
    
    Args:
        predictions: List of predictions
        model_name: Name of the model used
        output_dir: Directory to save predictions
        
    Returns:
        Path to the saved file
    """
    import os
    
    output_dir = output_dir or config.PREDICTIONS_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model_name = model_name.replace("/", "_")
    filename = f"{safe_model_name}_{timestamp}.json"
    output_path = os.path.join(output_dir, filename)
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Predictions saved to {output_path}")
        return output_path
        
    except Exception as e:
        raise RuntimeError(f"Failed to save predictions: {e}")


def main():
    """Main prediction function."""
    parser = argparse.ArgumentParser(
        description='Generate predictions using a fine-tuned model for Kotlin code completion.'
    )
    parser.add_argument(
        '--model_name', 
        required=True,
        help='Name of the base model'
    )
    parser.add_argument(
        '--model_path', 
        help='Path to custom model weights (optional)'
    )
    parser.add_argument(
        '--test_size', 
        type=int,
        help='Number of test examples to use (default: use all)'
    )
    parser.add_argument(
        '--output_dir',
        default=config.PREDICTIONS_DIR,
        help=f'Directory to save predictions (default: {config.PREDICTIONS_DIR})'
    )
    parser.add_argument(
        '--token_limit',
        type=int,
        default=config.DEFAULT_TOKEN_LIMIT,
        help=f'Maximum number of input tokens (default: {config.DEFAULT_TOKEN_LIMIT})'
    )
    parser.add_argument(
        '--max_new_tokens',
        type=int,
        default=config.DEFAULT_MAX_NEW_TOKENS,
        help=f'Maximum number of new tokens to generate (default: {config.DEFAULT_MAX_NEW_TOKENS})'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=config.DEFAULT_TEMPERATURE,
        help=f'Sampling temperature (default: {config.DEFAULT_TEMPERATURE})'
    )
    args = parser.parse_args()

    logger.info("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(args.model_name, args.model_path)

    logger.info("Loading test dataset...")
    dataset = load_kt_dataset()
    test_data = dataset['test']
    
    # Limit test size if specified
    if args.test_size:
        test_data = test_data.select(range(min(args.test_size, len(test_data))))
    
    logger.info(f"Generating predictions for {len(test_data)} examples...")
    
    predictions = []
    for example in tqdm(test_data, desc="Generating predictions"):
        if example['input'] and example['input'].strip():
            pred = single_predict(
                model, 
                tokenizer, 
                example['input'],
                token_limit=args.token_limit,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature
            )
            predictions.append(pred)
        else:
            predictions.append("")  # Empty prediction for empty input

    logger.info(f"Generated {len(predictions)} predictions")
    
    # Save predictions
    output_path = save_predictions(predictions, args.model_name, args.output_dir)
    logger.info(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    main()
