from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer,DataCollatorForSeq2Seq, TrainerCallback
from torch.utils.tensorboard import SummaryWriter
import argparse
from utils import load_kt_dataset
from functools import partial

WEIGHTS_DIR = 'weights/'


class TensorBoardCallback(TrainerCallback):
    def __init__(self, writer: SummaryWriter):
        self.writer = writer

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            for key, value in logs.items():
                self.writer.add_scalar(f"trainer/{key}", value, state.epoch)


def preprocess_function(tokenizer, examples):
    model_inputs = tokenizer(examples["input"], truncation=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["input"], truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def main():
    parser = argparse.ArgumentParser(description='Fine-tuner for code completion on custom Kotlin dataset.')
    parser.add_argument('--model_name', type=str, default='microsoft/phi-1_5', help='Base model for fine-tuning.')
    parser.add_argument('--train_batch_size', type=int, default=2)
    parser.add_argument('--eval_batch_size', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--num_train_epochs', type=int, default=3)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2)
    args = parser.parse_args()

    ds_kt = load_kt_dataset()

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    tokenized_datasets = ds_kt.map(partial(preprocess_function, tokenizer), batched=True)

    writer = SummaryWriter()

    training_args = Seq2SeqTrainingArguments(
        output_dir=WEIGHTS_DIR,
        predict_with_generate=True,
        evaluation_strategy="epoch",
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        save_steps=10_000,
        num_train_epochs=args.num_training_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
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
