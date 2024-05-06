from parsers import *

if __name__ == '__main__':
    kt_files_directory = "data/kotlin-master"
    kt_output_dataset_file = "data/kt_code_completion.json"

    parse_kt_dataset(kt_files_directory, kt_output_dataset_file)

    py_files_directory = "data/py150.jsonl"
    py_output_dataset_file = "data/py_code_completion.json"

    parse_py_dataset(py_files_directory, py_output_dataset_file)
