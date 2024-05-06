import json
import random
from tqdm import tqdm
import os
import argparse
from datasets import load_dataset
import subprocess

PY_DS_PATH = 'data/py_ds.json'
KT_DS_PATH = 'data/kt_ds.json'
KT_FILES_DIR = 'data/kt-master'
KT_REPO_LINK = 'https://github.com/jetbrains/kotlin.git'
DATA_FOLDER = 'data'


def clone_github_repo(repo_url, destination_path):
    try:
        subprocess.run(["git", "clone", repo_url, destination_path], check=True)
        print("Repository cloned successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Failed to clone repository: {e}")


def parse_kt_dataset(output_file):
    if os.path.exists(output_file):
        print('Kotlin data already parsed!')
    else:
        print('Parsing Kotlin data!')
        if True:
            clone_github_repo(KT_REPO_LINK, DATA_FOLDER)

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
    if os.path.exists(output_file):
        print('Python data already parsed!')
    else:
        print('Parsing Python data!')
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
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--type', required=True, help="Type of dataset to be parsed [python|kotlin]")
    args = parser.parse_args()

    assert args.type in ['python', 'kotlin']

    if args.type == 'kotlin':
        parse_kt_dataset(KT_DS_PATH)
    else:
        parse_py_dataset(PY_DS_PATH)


if __name__ == '__main__':
    main()
