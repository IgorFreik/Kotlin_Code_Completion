from tqdm import tqdm
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

