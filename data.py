# import re
# from typing import Dict, Tuple

# from datasets import DatasetDict, load_dataset


# TASK_TO_TEXT = {
#     "sst2": {
#         "input_columns": ("sentence",),
#         "label_map": {0: "negative", 1: "positive"},
#         "prompt_fn": lambda example: f"sst2 sentence: {example['sentence']} sentiment:",
#     },
#     "mrpc": {
#         "input_columns": ("sentence1", "sentence2"),
#         "label_map": {0: "not_equivalent", 1: "equivalent"},
#         "prompt_fn": lambda example: (
#             f"mrpc sentence1: {example['sentence1']} sentence2: {example['sentence2']} equivalent:"
#         ),
#     },
#     "qnli": {
#         "input_columns": ("question", "sentence"),
#         "label_map": {0: "entailment", 1: "not_entailment"},
#         "prompt_fn": lambda example: (
#             f"qnli question: {example['question']} sentence: {example['sentence']} answer:"
#         ),
#     },
# }


# def normalize_text(value: str) -> str:
#     value = value.strip().lower()
#     value = re.sub(r"\s+", " ", value)
#     return value


# def _build_text_to_text_dataset(
#     raw_datasets: DatasetDict,
#     tokenizer,
#     task_name: str,
#     max_source_length: int,
#     max_target_length: int,
# ) -> DatasetDict:
#     task_spec = TASK_TO_TEXT[task_name]
#     active_splits = DatasetDict(
#         {
#             split_name: raw_datasets[split_name]
#             for split_name in ("train", "validation")
#             if split_name in raw_datasets
#         }
#     )

#     def preprocess(example: Dict) -> Dict:
#         model_input = task_spec["prompt_fn"](example)
#         target_text = task_spec["label_map"][int(example["label"])]
#         tokenized_inputs = tokenizer(
#             model_input,
#             max_length=max_source_length,
#             truncation=True,
#         )
#         tokenized_targets = tokenizer(
#             text_target=target_text,
#             max_length=max_target_length,
#             truncation=True,
#         )
#         return {
#             "input_ids": tokenized_inputs["input_ids"],
#             "attention_mask": tokenized_inputs["attention_mask"],
#             "labels": tokenized_targets["input_ids"],
#         }

#     processed = active_splits.map(
#         preprocess,
#         remove_columns=active_splits["train"].column_names,
#         desc=f"Tokenizing {task_name}",
#     )
#     return processed


# def load_glue_text2text(
#     tokenizer,
#     task_name: str = "sst2",
#     max_source_length: int = 192,
#     max_target_length: int = 8,
#     max_train_samples: int = 0,
#     max_eval_samples: int = 0,
# ) -> Tuple[object, object]:
#     if task_name not in TASK_TO_TEXT:
#         raise ValueError(f"Unsupported task '{task_name}'. Choose from: {sorted(TASK_TO_TEXT)}")

#     raw_datasets = load_dataset("glue", task_name)
#     processed = _build_text_to_text_dataset(
#         raw_datasets=raw_datasets,
#         tokenizer=tokenizer,
#         task_name=task_name,
#         max_source_length=max_source_length,
#         max_target_length=max_target_length,
#     )

#     train_dataset = processed["train"]
#     eval_dataset = processed["validation"]

#     if max_train_samples and max_train_samples > 0:
#         train_dataset = train_dataset.select(range(min(max_train_samples, len(train_dataset))))
#     if max_eval_samples and max_eval_samples > 0:
#         eval_dataset = eval_dataset.select(range(min(max_eval_samples, len(eval_dataset))))

#     return train_dataset, eval_dataset

from typing import Tuple

from datasets import load_dataset

def normalize_text(value: str) -> str:
    import re
    value = value.strip().lower()
    value = re.sub(r"\s+", " ", value)
    return value

def load_xsum_text2text(
    tokenizer,
    max_source_length: int = 512,
    max_target_length: int = 128,
    max_train_samples: int = 0,
    max_eval_samples: int = 0,
) -> Tuple[object, object]:
    raw = load_dataset("xsum")

    def preprocess(example):
        inputs = tokenizer(
            "summarize: " + example["document"],
            max_length=max_source_length,
            truncation=True,
        )
        targets = tokenizer(
            text_target=example["summary"],
            max_length=max_target_length,
            truncation=True,
        )
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": targets["input_ids"],
        }

    processed = raw.map(
        preprocess,
        remove_columns=raw["train"].column_names,
        desc="Tokenizing xsum",
    )

    train_dataset = processed["train"]
    eval_dataset = processed["validation"]

    if max_train_samples and max_train_samples > 0:
        train_dataset = train_dataset.select(range(min(max_train_samples, len(train_dataset))))
    if max_eval_samples and max_eval_samples > 0:
        eval_dataset = eval_dataset.select(range(min(max_eval_samples, len(eval_dataset))))

    return train_dataset, eval_dataset
