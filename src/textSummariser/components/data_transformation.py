import os

from datasets import load_from_disk
from transformers import AutoTokenizer

from src.textSummariser.entity import DataTransformationConfig


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    def convert_examples_to_features(self, example_batch):
        # Ensure we have valid text data
        dialogues = example_batch["dialogue"]
        summaries = example_batch["summary"]

        # Convert to list of strings if needed and filter out None/empty values
        if not isinstance(dialogues, list):
            dialogues = [dialogues] if dialogues else [""]
        if not isinstance(summaries, list):
            summaries = [summaries] if summaries else [""]

        # Clean the data - ensure all entries are strings
        dialogues = [str(d) if d is not None else "" for d in dialogues]
        summaries = [str(s) if s is not None else "" for s in summaries]

        input_encodings = self.tokenizer(
            dialogues,
            max_length=1024,
            truncation=True,
            padding=True,
            return_tensors=None,
        )

        with self.tokenizer.as_target_tokenizer():
            target_encodings = self.tokenizer(
                summaries,
                max_length=128,
                truncation=True,
                padding=True,
                return_tensors=None,
            )

        return {
            "input_ids": input_encodings["input_ids"],
            "attention_mask": input_encodings["attention_mask"],
            "labels": target_encodings["input_ids"],
        }

    def convert(self):
        dataset_samsum = load_from_disk(self.config.data_path)
        dataset_samsum_pt = dataset_samsum.map(
            self.convert_examples_to_features, batched=True
        )
        dataset_samsum_pt.save_to_disk(
            os.path.join(self.config.root_dir, "samsum_dataset")
        )
