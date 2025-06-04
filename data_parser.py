import argparse
import json
import logging
import os
import random
import zipfile
from typing import List, Dict, Any

import requests
from datasets import load_dataset
from tqdm import tqdm

import config

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))


def download_github_repo(repo_owner: str, repo_name: str, destination_path: str) -> None:
    """
    Download a GitHub repository as a ZIP file and extract it to a local path.
    
    Args:
        repo_owner: The GitHub repository owner
        repo_name: The GitHub repository name
        destination_path: The local path to store the extracted repository
    
    Raises:
        RuntimeError: If the download fails
    """
    zip_url = f"https://github.com/{repo_owner}/{repo_name}/archive/refs/heads/master.zip"
    
    try:
        response = requests.get(zip_url, timeout=30)
        response.raise_for_status()
        
        os.makedirs(destination_path, exist_ok=True)
        
        zip_file_path = os.path.join(destination_path, f"{repo_name}.zip")
        with open(zip_file_path, "wb") as zip_file:
            zip_file.write(response.content)
        
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall(destination_path)
        
        os.remove(zip_file_path)
        logger.info("Successfully downloaded the repository!")
        
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to download repository: {e}")
    except zipfile.BadZipFile as e:
        raise RuntimeError(f"Invalid ZIP file: {e}")


def _process_kotlin_file(file_path: str) -> List[Dict[str, str]]:
    """
    Process a single Kotlin file and extract training examples.
    
    Args:
        file_path: Path to the Kotlin file
        
    Returns:
        List of dictionaries containing input-label pairs
    """
    examples = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f_in:
            lines = f_in.readlines()
            
            if not lines:
                return examples
                
            line_index = random.randint(0, len(lines) - 1)
            prefix = ''.join(lines[:line_index])
            postfix = lines[line_index]
            
            if len(prefix.strip()) > 0 and len(postfix.strip()) > 0:
                examples.append({
                    'input': prefix.strip(),
                    'labels': postfix.strip()
                })
                
    except (UnicodeDecodeError, IOError) as e:
        logger.warning(f"Failed to process file {file_path}: {e}")
        
    return examples


def parse_kt_dataset(output_file: str) -> None:
    """
    Parse a Kotlin dataset from the JetBrains/Kotlin GitHub repository
    and prepare it for code completion tasks.
    
    Args:
        output_file: The file path to store the processed dataset
    """
    if os.path.exists(output_file):
        logger.info('Kotlin data already parsed. Skipping.')
        return
        
    logger.info('Kotlin dataset not found. Starting parsing.')
    
    if not os.path.exists(config.KT_FILES_DIR):
        download_github_repo(config.KT_REPO_OWNER, config.KT_REPO_NAME, config.DATA_FOLDER)
    
    random.seed(config.RANDOM_SEED)
    dataset = []
    
    for root, _, files in tqdm(os.walk(config.KT_FILES_DIR), desc="Processing Kotlin files"):
        for file in files:
            if any(file.endswith(ext) for ext in config.KOTLIN_EXTENSIONS):
                file_path = os.path.join(root, file)
                examples = _process_kotlin_file(file_path)
                dataset.extend(examples)
    
    logger.info(f"Processed {len(dataset)} examples from Kotlin files")
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f_out:
            json.dump(dataset, f_out, indent=2, ensure_ascii=False)
        logger.info(f"Kotlin dataset saved to {output_file}")
    except IOError as e:
        raise RuntimeError(f"Failed to save Kotlin dataset: {e}")


def parse_py_dataset(output_file: str) -> None:
    """
    Download the Python dataset (microsoft/code_method_completion) and prepare it 
    for code completion tasks.
    
    Args:
        output_file: The file path to store the processed dataset
    """
    if os.path.exists(output_file):
        logger.info('Python data already parsed. Skipping.')
        return
        
    logger.info('Python dataset not found. Starting parsing.')
    random.seed(config.RANDOM_SEED)
    
    try:
        ds = load_dataset('microsoft/code_method_completion')
        result = []
        
        for method in tqdm(ds['train'], desc="Processing Python methods"):
            lines = method['body'].split('<EOL>')
            lines = [line for line in lines if line.strip()]
            
            if not lines:
                continue
                
            rnd_idx = random.randint(0, len(lines) - 1)
            body_prefix = '\n'.join(lines[:rnd_idx])
            prefix = f'{method["signature"]}\n"""{method["docstring"]}"""\n{body_prefix}\n'
            suffix = lines[rnd_idx]
            
            result.append({
                'input': prefix,
                'labels': suffix
            })
        
        logger.info(f"Processed {len(result)} examples from Python dataset")
        
        with open(output_file, 'w', encoding='utf-8') as f_out:
            json.dump(result, f_out, indent=2, ensure_ascii=False)
        logger.info(f"Python dataset saved to {output_file}")
        
    except Exception as e:
        raise RuntimeError(f"Failed to process Python dataset: {e}")


def main():
    """Main function to parse datasets based on command line arguments."""
    parser = argparse.ArgumentParser(
        description='Parse datasets for code completion training.'
    )
    parser.add_argument(
        '--type', 
        required=True, 
        nargs='+', 
        choices=['python', 'kotlin'],
        help="Type of dataset to be parsed"
    )
    args = parser.parse_args()
    
    for ds_type in args.type:
        if ds_type == 'kotlin':
            parse_kt_dataset(config.KT_DS_PATH)
        elif ds_type == 'python':
            parse_py_dataset(config.PY_DS_PATH)


if __name__ == '__main__':
    main()
