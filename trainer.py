import argparse
import logging
from functools import partial
from typing import Dict, Any

import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq, 
    TrainerCallback
)

import config
from utils import load_kt_dataset

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL), format=config.LOG_FORMAT)


class TensorBoardCallback(TrainerCallback):
    """Custom callback for logging training metrics to TensorBoard."""
    
    def __init__(self, writer: SummaryWriter):
        """
        Initialize the TensorBoard callback.
        
        Args:
            writer: TensorBoard SummaryWriter instance
        """
        self.writer = writer

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log training metrics to TensorBoard."""
        if logs:
            for key, value in logs.items():
                self.writer.add_scalar(f"trainer/{key}", value, state.global_step)


def preprocess_function(tokenizer: AutoTokenizer, examples: Dict[str, Any]) -> Dict[str, Any]:
    """
    Preprocess examples for training by tokenizing inputs and labels.
    
    Args:
        tokenizer: The tokenizer to use for preprocessing
        examples: Dictionary containing input examples
        
    Returns:
        Dictionary with tokenized inputs and labels
    """
    # Tokenize inputs
    model_inputs = tokenizer(examples["input"], truncation=True, padding=True)
    
    # Tokenize labels
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["labels"], truncation=True, padding=True)
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def setup_model_and_tokenizer(model_name: str) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load and setup the model and tokenizer.
    
    Args:
        model_name: Name of the pre-trained model to use
        
    Returns:
        Tuple of (model, tokenizer)
    """
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add padding token if it doesn't exist
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))
            
        return model, tokenizer
    except Exception as e:
        raise RuntimeError(f"Failed to load model {model_name}: {e}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description='Fine-tune a model for Kotlin code completion.'
    )
    parser.add_argument(
        '--model_name', 
        type=str, 
        default=config.DEFAULT_MODEL_NAME,
        help='Base model for fine-tuning'
    )
    parser.add_argument(
        '--train_batch_size', 
        type=int, 
        default=config.DEFAULT_TRAIN_BATCH_SIZE,
        help='Training batch size'
    )
    parser.add_argument(
        '--eval_batch_size', 
        type=int, 
        default=config.DEFAULT_EVAL_BATCH_SIZE,
        help='Evaluation batch size'
    )
    parser.add_argument(
        '--learning_rate', 
        type=float, 
        default=config.DEFAULT_LEARNING_RATE,
        help='Learning rate'
    )
    parser.add_argument(
        '--num_train_epochs', 
        type=int, 
        default=config.DEFAULT_NUM_EPOCHS,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--gradient_accumulation_steps', 
        type=int, 
        default=config.DEFAULT_GRADIENT_ACCUMULATION_STEPS,
        help='Gradient accumulation steps'
    )
    args = parser.parse_args()

    logger.info("Loading dataset...")
    ds_kt = load_kt_dataset()
    logger.info(f"Dataset loaded with {len(ds_kt['train'])} training examples")

    logger.info(f"Loading model and tokenizer: {args.model_name}")
    model, tokenizer = setup_model_and_tokenizer(args.model_name)

    logger.info("Tokenizing dataset...")
    tokenized_datasets = ds_kt.map(
        partial(preprocess_function, tokenizer), 
        batched=True,
        desc="Tokenizing"
    )

    # Setup TensorBoard logging
    writer = SummaryWriter(log_dir=config.LOGS_DIR)

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=config.WEIGHTS_DIR,
        predict_with_generate=True,
        evaluation_strategy="epoch",
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        save_steps=10_000,
        num_train_epochs=args.num_train_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_dir=config.LOGS_DIR,
        report_to="tensorboard",
        save_total_limit=2,  # Keep only the best 2 checkpoints
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    # Initialize trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["eval"],
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        callbacks=[TensorBoardCallback(writer)],
    )

    logger.info("Starting training...")
    try:
        trainer.train()
        logger.info("Training completed successfully!")
        
        # Save the final model
        trainer.save_model()
        tokenizer.save_pretrained(config.WEIGHTS_DIR)
        logger.info(f"Model saved to {config.WEIGHTS_DIR}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        writer.close()


if __name__ == "__main__":
    main()
