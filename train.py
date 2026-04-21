import argparse
import inspect
import os
from typing import Dict

import numpy as np
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    T5ForConditionalGeneration,
    set_seed,
)

from data import load_glue_text2text, normalize_text


def build_training_args(args):
    signature = inspect.signature(Seq2SeqTrainingArguments.__init__)
    valid_keys = set(signature.parameters.keys())

    common_kwargs = {
        "output_dir": args.output_dir,
        "do_train": True,
        "do_eval": True,
        "predict_with_generate": True,
        "generation_max_length": args.generation_max_length,
        "remove_unused_columns": False,
        "per_device_train_batch_size": args.per_device_batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "optim": args.optim,
        "num_train_epochs": args.epochs,
        "logging_steps": args.logging_steps,
        "save_total_limit": args.save_total_limit,
        "dataloader_num_workers": args.num_workers,
        "dataloader_pin_memory": True,
        "report_to": ["wandb"] if args.wandb_project else [],
        "bf16": args.bf16,
        "fp16": args.fp16,
        "gradient_checkpointing": args.gradient_checkpointing,
        "deepspeed": args.deepspeed_config if args.deepspeed_config else None,
        "fsdp": "full_shard auto_wrap" if args.fsdp else "",
        "fsdp_config": args.fsdp_config if args.fsdp_config else None,
        "ddp_find_unused_parameters": False,
    }

    if "eval_strategy" in valid_keys:
        common_kwargs["eval_strategy"] = args.eval_strategy
    elif "evaluation_strategy" in valid_keys:
        common_kwargs["evaluation_strategy"] = args.eval_strategy

    if "save_strategy" in valid_keys:
        common_kwargs["save_strategy"] = args.save_strategy

    filtered_kwargs = {
        key: value
        for key, value in common_kwargs.items()
        if key in valid_keys and value is not None
    }
    return Seq2SeqTrainingArguments(**filtered_kwargs)


def build_trainer(model, training_args, train_dataset, eval_dataset, tokenizer, data_collator, compute_metrics):
    signature = inspect.signature(Seq2SeqTrainer.__init__)
    valid_keys = set(signature.parameters.keys())
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
        "compute_metrics": compute_metrics,
    }

    # Older/newer Transformers versions disagree on whether this should be
    # passed as `tokenizer`, `processing_class`, or omitted entirely.
    if "tokenizer" in valid_keys:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in valid_keys:
        trainer_kwargs["processing_class"] = tokenizer

    filtered_kwargs = {key: value for key, value in trainer_kwargs.items() if key in valid_keys}
    return Seq2SeqTrainer(**filtered_kwargs)


def parse_args():
    parser = argparse.ArgumentParser("Library-based T5 Large distributed fine-tuning")
    parser.add_argument("--model-name", type=str, default="t5-large")
    parser.add_argument("--task-name", type=str, default="sst2", choices=["sst2", "mrpc", "qnli"])
    parser.add_argument("--output-dir", type=str, default=os.path.join(os.path.dirname(__file__), "outputs"))
    parser.add_argument("--deepspeed-config", type=str, default="")
    parser.add_argument("--fsdp-config", type=str, default="")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--per-device-batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--optim", type=str, default="adafactor")
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-strategy", type=str, default="epoch")
    parser.add_argument("--eval-strategy", type=str, default="epoch")
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--max-source-length", type=int, default=192)
    parser.add_argument("--max-target-length", type=int, default=8)
    parser.add_argument("--generation-max-length", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--fsdp", action="store_true")
    parser.add_argument("--resume-from-checkpoint", type=str, default="")
    # W&B tracking
    parser.add_argument("--wandb-entity", type=str, default="t5_mlsys")
    parser.add_argument("--wandb-project", type=str, default="deepseed")
    parser.add_argument("--wandb-run-name", type=str, default="")
    return parser.parse_args()


def build_compute_metrics(tokenizer):
    def compute_metrics(eval_pred) -> Dict[str, float]:
        predictions, labels = eval_pred

        if isinstance(predictions, tuple):
            predictions = predictions[0]

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        pred_texts = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        label_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)

        normalized_preds = [normalize_text(text) for text in pred_texts]
        normalized_labels = [normalize_text(text) for text in label_texts]
        accuracy = float(
            sum(int(pred == label) for pred, label in zip(normalized_preds, normalized_labels))
            / max(1, len(normalized_labels))
        )
        return {"accuracy": accuracy}

    return compute_metrics


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.wandb_project:
        os.environ["WANDB_ENTITY"] = args.wandb_entity
        os.environ["WANDB_PROJECT"] = args.wandb_project
        if args.wandb_run_name:
            os.environ["WANDB_NAME"] = args.wandb_run_name

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    train_dataset, eval_dataset = load_glue_text2text(
        tokenizer=tokenizer,
        task_name=args.task_name,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8 if (args.fp16 or args.bf16) else None,
    )

    training_args = build_training_args(args)

    trainer = build_trainer(
        model=model,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(tokenizer),
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint or None)
    metrics = trainer.evaluate(metric_key_prefix="eval")
    trainer.save_model(os.path.join(args.output_dir, "final_model"))
    trainer.save_metrics("eval", metrics)
    trainer.save_state()


if __name__ == "__main__":
    main()
