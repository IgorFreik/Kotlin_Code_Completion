import json
import random
from tqdm import tqdm
import argparse
from datasets import load_dataset
import logging
import requests
import os
import zipfile

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


PY_DS_PATH = 'data/py_ds.json'
KT_DS_PATH = 'data/kt_ds.json'
KT_FILES_DIR = 'data/kotlin-master'
KT_REPO_NAME = 'kotlin'
KT_REPO_OWNER = 'jetbrains'
DATA_FOLDER = 'data'


def download_github_repo(repo_owner, repo_name, destination_path):
    """
    Function to clone a github repository to a given local path given it's web path.
    :param repo_owner: The name of the github repo owner.
    :param repo_name: The name of the github repo.
    :param destination_path: The local path to store the cloned repo.
    """
    zip_url = f"https://github.com/{repo_owner}/{repo_name}/archive/refs/heads/master.zip"
    response = requests.get(zip_url)
    if response.status_code == 200:
        os.makedirs(destination_path, exist_ok=True)

        zip_file_path = os.path.join(destination_path, f"{repo_name}.zip")
        with open(zip_file_path, "wb") as zip_file:
            zip_file.write(response.content)

        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall(destination_path)
        os.remove(zip_file_path)
        print("Successfully downloaded the repo!")
    else:
        print("Failed to download repository!")


def parse_kt_dataset(output_file):
    """
    Function to parse a Kotlin dataset from the jetbrains/Kotlin github repository
    and prepare it for the task of code completion.
    :param output_file: The string file path to store the downloaded dataset.
    """
    if os.path.exists(output_file):
        logger.info('Kotlin data already parsed. Aborting.')
    else:
        logger.info('Kotlin dataset not found. Starting parsing.')
        if not os.path.exists(KT_FILES_DIR):
            download_github_repo(KT_REPO_NAME, KT_REPO_OWNER, DATA_FOLDER)

        random.seed(42)
        dataset = []

        for root, _, files in tqdm(os.walk(KT_FILES_DIR)):
            for file in files:
                if file.endswith('.kt') or file.endswith('.kts'):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r', encoding='utf-8') as f_in:
                        lines = f_in.readlines()
                        if lines:
                            line_index = random.randint(0, len(lines) - 1)
                            prefix = ''.join(lines[:line_index])
                            postfix = lines[line_index]
                            if len(prefix.strip()) * len(postfix.strip()) > 0:
                                dataset.append({'input': prefix.strip(), 'labels': postfix.strip()})

        with open(output_file, 'w', encoding='utf-8') as f_out:
            json.dump(dataset, f_out, indent=2)


def parse_py_dataset(output_file):
    """
    Function to download the python dataset PY150 and prepare it for the task of code completion.
    :param output_file: The string file path to store the downloaded dataset.
    """
    if os.path.exists(output_file):
        logger.info('Python data already parsed. Aborting.')
    else:
        logger.info('Python dataset not found. Starting parsing.')
        random.seed(42)

        ds = load_dataset('microsoft/code_method_completion')
        result = []

        for method in ds:
            lines = method['body'].split('<EOL>')
            lines = [l for l in lines if l != '']

            rnd_idx = random.randint(0, len(lines) - 1)

            body_prefix = '\n'.join(lines[:rnd_idx])
            prefix = f'{method["signature"]}\n"""{method["docstring"]}"""\n{body_prefix}\n'
            suffix = lines[rnd_idx]

            result.append({'input': prefix, 'labels': suffix})

        with open(output_file, 'w', encoding='utf-8') as f_out:
            json.dump(result, f_out, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Parser to fetch data from Py150 dataset and scrape a custom Kotlin dataset.')
    parser.add_argument('--type', required=True, nargs='+', help="Type of dataset to be parsed [python|kotlin]")
    args = parser.parse_args()

    for ds_type in args.type:
        assert ds_type in ['python', 'kotlin']

        if ds_type == 'kotlin':
            parse_kt_dataset(KT_DS_PATH)
        else:
            parse_py_dataset(PY_DS_PATH)


if __name__ == '__main__':
    main()
