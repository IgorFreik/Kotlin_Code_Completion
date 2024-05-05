# Code adapted from here:
# https://github.com/microsoft/CodeXGLUE/blob/main/Code-Code/CodeCompletion-line/evaluator/evaluator.py

from tqdm import tqdm
from fuzzywuzzy import fuzz
import numpy as np


def single_predict(model, tokenizer, input, token_lim=1000):
    # TODO: Add postprecessing

    tokens = tokenizer(input, return_tensors="pt", return_attention_mask=False)
    input_len = len(tokens['input_ids'][0])

    if input_len > token_lim:
        tokens['input_ids'] = tokens['input_ids'][0][-token_lim:].reshape(1, -1)
        input_len = token_lim

    outputs = model.generate(**tokens, max_length=input_len+50)
    pred = tokenizer.decode(outputs[0][input_len:])
    pred = pred.split('\n')

    return pred[0]


def evaluate(model, tokenizer, dataset, test_size=500):
    results = {'EM': [], 'edit_sim': []}

    for example in tqdm(dataset[:test_size]):
        if example['input']:
            pred = single_predict(model, tokenizer, example['input'])
            gt = example['gt']

            edit_res = fuzz.ratio(pred, gt)
            em_res = int(pred.split() == gt.split())

            results['edit_sim'].append(edit_res)
            results['EM'].append(em_res)

    print(f'Mean Levenshtein distance: {np.array(results["edit_sim"]).mean()}.')
    print(f'Mean EM: {np.array(results["EM"]).mean()}.')
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate leaderboard predictions for code completion (line level).')
    parser.add_argument('--answers', '-a', required=True, help="filename of the labels, in json format.")
    parser.add_argument('--predictions', '-p', required=True, help="filename of the leaderboard predictions, in txt format.")
    args = parser.parse_args()

    preds = open(args.predictions, "r").readlines()
    gts = open(args.answers, "r").readlines()

    assert len(preds) == len(gts), f"Samples of predictions and answers are not equal, {len(preds)}: {len(gts)}"

    total = len(gts)
    EM = 0.0
    edit_sim = 0.0
    for pred, gt in zip(preds, gts):
        pred = post_process(pred.strip())
        gt = post_process(json.loads(gt)["gt"])
        edit_sim += fuzz.ratio(pred, gt)
        if pred.split() == gt.split():
            EM += 1

    logger.info(f"Edit sim: {round(edit_sim/total, 2)}, EM: {round(EM/total*100, 2)}")


if __name__ == "__main__":
    main()
