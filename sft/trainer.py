from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer,DataCollatorForSeq2Seq, TrainerCallback
from torch.utils.tensorboard import SummaryWriter
import json
import os


class TensorBoardCallback(TrainerCallback):
    def __init__(self, writer: SummaryWriter):
        self.writer = writer

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            for key, value in logs.items():
                self.writer.add_scalar(f"trainer/{key}", value, state.epoch)


def load_kt_dataset():
    ds_kt = json.load(open(os.path.join(os.getcwd(), '../data/kt_code_completion.json')))
    ds_kt = load_dataset('json', data_files='./kt_code_completion_dataset.json')['train'].rename_column('target',
                                                                                                        'labels').shard(
        num_shards=40, index=0)
    ds_kt = ds_kt.train_test_split(test_size=0.3)

    ds_kt_remaining = ds_kt['test'].train_test_split(test_size=0.5)

    ds_kt['eval'] = ds_kt_remaining['train']
    ds_kt['test'] = ds_kt_remaining['test']
    return ds_kt


def preprocess_function(examples):
    model_inputs = tokenizer(examples["input"], truncation=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["input"], truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_name', required=True)
    args = parser.parse_args()

    ds_kt = load_kt_dataset()

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    tokenized_datasets = ds_kt.map(preprocess_function, batched=True)

    writer = SummaryWriter()

    training_args = Seq2SeqTrainingArguments(
        output_dir='./output_dir',
        predict_with_generate=True,
        evaluation_strategy="epoch",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=4,
        learning_rate=1e5,
        save_steps=10_000,
        save_total_limit=2,
        num_train_epochs=3,
        logging_dir="./logs",
        report_to="tensorboard",

    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["eval"],
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        callbacks=[TensorBoardCallback(writer)],
    )

    trainer.train()


if __name__ == "__main__":
    main()
