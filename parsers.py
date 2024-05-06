import json
import random
from tqdm import tqdm
import os


def read_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            data.append(json.loads(line))
    return data


def parse_kt_dataset(kotlin_files_dir, output_file):
    random.seed(42)
    dataset = []
    for root, _, files in tqdm(os.walk(kotlin_files_dir)):
        for file in files:
            if file.endswith('.kt') or file.endswith('.kts'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f_in:
                    lines = f_in.readlines()
                    if lines:  # Ensure file is not empty
                        line_index = random.randint(0, len(lines) - 1)
                        prefix = ''.join(lines[:line_index])
                        postfix = lines[line_index]
                        if len(prefix.strip()) * len(postfix.strip()) > 0:
                            dataset.append({'input': prefix.strip(), 'labels': postfix.strip()})

    with open(output_file, 'w', encoding='utf-8') as f_out:
        json.dump(dataset, f_out, indent=2)


def parse_py_dataset(python_file_dir, output_file):
    """
    Parsing function for the python dataset.
    """
    random.seed(42)

    ds = read_jsonl(python_file_dir)
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
