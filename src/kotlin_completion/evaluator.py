# Code taken from this source:
# https://github.com/microsoft/CodeXGLUE/blob/main/Code-Code/CodeCompletion-line/evaluator/evaluator.py

import argparse
import json
import logging
import re
from typing import List, Union

from fuzzywuzzy import fuzz

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def post_process(code: str) -> str:
    """
    Post-process code by replacing placeholder tokens with their actual values.
    
    Args:
        code: The code string to post-process
        
    Returns:
        The processed code string
    """
    if not code:
        return ""
    
    # Replace simple placeholders
    code = code.replace("<NUM_LIT>", "0").replace("<STR_LIT>", "").replace("<CHAR_LIT>", "")
    
    # Replace complex placeholders with regex
    pattern = re.compile(r"<(STR|NUM|CHAR)_LIT:(.*?)>", re.S)
    lits = re.findall(pattern, code)
    for lit in lits:
        code = code.replace(f"<{lit[0]}_LIT:{lit[1]}>", lit[1])
    
    return code


def load_data(file_path: str, file_type: str = 'txt') -> List[str]:
    """
    Load data from a file, supporting both txt and json formats.
    
    Args:
        file_path: Path to the file
        file_type: Type of file ('txt' or 'json')
        
    Returns:
        List of strings
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_type == 'json':
                data = json.load(f)
                # Handle both list of strings and list of dicts
                if isinstance(data, list):
                    if data and isinstance(data[0], dict):
                        # Assume it's a list of prediction objects
                        return [str(item.get('prediction', '')) for item in data]
                    else:
                        return [str(item) for item in data]
                else:
                    return [str(data)]
            else:
                return [line.strip() for line in f.readlines()]
    except Exception as e:
        raise RuntimeError(f"Failed to load data from {file_path}: {e}")


def calculate_metrics(predictions: List[str], ground_truths: List[str]) -> dict:
    """
    Calculate evaluation metrics for code completion.
    
    Args:
        predictions: List of predicted strings
        ground_truths: List of ground truth strings
        
    Returns:
        Dictionary containing calculated metrics
    """
    if len(predictions) != len(ground_truths):
        raise ValueError(
            f"Number of predictions ({len(predictions)}) and ground truths ({len(ground_truths)}) must be equal"
        )
    
    total = len(ground_truths)
    exact_match = 0
    edit_similarity = 0.0
    
    for pred, gt in zip(predictions, ground_truths):
        # Post-process both prediction and ground truth
        pred_processed = post_process(pred.strip())
        gt_processed = post_process(gt.strip())
        
        # Calculate edit similarity (fuzzy matching score)
        edit_similarity += fuzz.ratio(pred_processed, gt_processed)
        
        # Calculate exact match (token-level comparison)
        if pred_processed.split() == gt_processed.split():
            exact_match += 1
    
    return {
        'edit_similarity': round(edit_similarity / total, 2),
        'exact_match': round(exact_match / total * 100, 2),
        'total_samples': total
    }


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description='Evaluate predictions for code completion (line level).'
    )
    parser.add_argument(
        '--answers', 
        '-a', 
        required=True, 
        help="Path to file containing ground truth labels"
    )
    parser.add_argument(
        '--predictions', 
        '-p', 
        required=True, 
        help="Path to file containing predictions"
    )
    parser.add_argument(
        '--prediction_format',
        choices=['txt', 'json'],
        default='txt',
        help="Format of the prediction file (default: txt)"
    )
    parser.add_argument(
        '--answer_format',
        choices=['txt', 'json'],
        default='txt',
        help="Format of the answer file (default: txt)"
    )
    args = parser.parse_args()

    logger.info("Loading predictions and ground truths...")
    
    try:
        predictions = load_data(args.predictions, args.prediction_format)
        ground_truths = load_data(args.answers, args.answer_format)
        
        logger.info(f"Loaded {len(predictions)} predictions and {len(ground_truths)} ground truths")
        
        # Calculate metrics
        metrics = calculate_metrics(predictions, ground_truths)
        
        # Display results
        logger.info("Evaluation Results:")
        logger.info(f"  Edit Similarity: {metrics['edit_similarity']}")
        logger.info(f"  Exact Match: {metrics['exact_match']}%")
        logger.info(f"  Total Samples: {metrics['total_samples']}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
