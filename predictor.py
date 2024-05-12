import argparse
import logging
from evaluator import post_process
from utils import load_kt_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import json
import torch
from datetime import datetime

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def single_predict(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, input: str, token_lim: int = 1000):
    """
    A function to make predictions given model, tokenizer, max tokens length and the input string.
    :param model: The model to make predicitons with.
    :param tokenizer: The tokenizer for the model to make predictins with.
    :param input: The input string with code prefix.
    :param token_lim: The max tokens to input into the model.
    :return: The predicted string (code line).
    """
    tokens = tokenizer(post_process(input), return_tensors="pt", return_attention_mask=False)
    input_len = len(tokens['input_ids'][0])

    if input_len > token_lim:
        tokens['input_ids'] = tokens['input_ids'][0][-token_lim:].reshape(1, -1)
        input_len = token_lim

    outputs = model.generate(**tokens, max_length=input_len+50)
    pred = tokenizer.decode(outputs[0][input_len:])
    pred = pred.split('\n')

    return pred[0]


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_name', required=True)
    parser.add_argument('--model_path', '-a', required=True, help="filename of the labels, in json format.")
    parser.add_argument('--test_size', '-a')
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    dataset = load_kt_dataset()
    preds = []

    for example in tqdm(dataset[:args.test_size]):
        if example['input']:
            preds.append(single_predict(model, tokenizer, example['input']))

    output_path = f'/weights/{args.model_name}_{datetime.now()}'
    json.dump(preds, open(output_path, 'w'))


if __name__ == "__main__":
    main()
