from pathlib import Path
from random import random

import ray
from datasets import load_dataset
from PIL import Image

from dataset.const import AnswerType, ProblemType

from .inference_engine import AsyncDPDispathEngineForThink


@ray.remote
class AsyncDPDispathEngineForSAT2(AsyncDPDispathEngineForThink):
    def __init__(self, dataset_path = None, dataset_split = None, data_files = None):
        dataset_path = dataset_path or "json"
        dataset_split = dataset_split or "train"
        data_files = data_files or "/mnt/data-alpha-sg-02/team-camera/huggingface/datasets/vigorl/spatial_reasoning/vigorl_sat2_train_RL.json"

        super().__init__(dataset_path, dataset_split, data_files)

    def preprocess(self, sample):
        processed_sample = {}
        processed_sample["source"] = "sat2"
        processed_sample["problem_type"] = ProblemType.SPATIAL_REASONING
        processed_sample["images"] = [Image.open(Path("/mnt/data-alpha-sg-02/team-camera/huggingface/datasets/vigorl/") / url) for url in sample.pop("image_paths")]

        problem = sample.pop("problem")
        problem = problem.replace("Question: ", "").replace("Answer with the text of the option.", "")

        image_num = problem.count("<image>")
        if image_num > len(processed_sample["images"]):
            # remove extra <image> tokens
            parts = problem.split("<image>")
            problem = "<image>".join(parts[: len(processed_sample["images"]) + 1])
        elif image_num < len(processed_sample["images"]):
            # add missing <image> tokens at the end
            problem = " <image>" * (len(processed_sample["images"]) - image_num) + "\n" + problem

        parts = problem.split("Answer Choices:")
        assert len(parts) == 2, f"Problem format error {problem}"

        question = parts[0].strip()
        choices = [x.strip()[3:] for x in parts[1].strip().split("\n")]

        answer = sample.pop("answer")
        assert answer in choices, f"Answer '{answer}' not in choices {choices}"
        option_index = choices.index(answer)
        option_letter = chr(ord("A") + option_index)

        if random() < 0.5:
            processed_sample["answer_type"] = AnswerType.MULTIPLE_CHOICE
            processed_sample["answer"] = option_letter
            choices_str = "\n".join([f"({chr(ord('A') + i)}) {choice}" for i, choice in enumerate(choices)])
            processed_sample["problem"] = f"{question}\nChoices:\n{choices_str}"
        else:
            processed_sample["problem"] = question
            processed_sample["answer"] = answer

            if len(answer) == 1 and answer in {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J"}:
                processed_sample["answer_type"] = AnswerType.MULTIPLE_CHOICE
            elif answer.lower() in {"true", "false", "yes", "no"}:
                processed_sample["answer_type"] = AnswerType.BOOLEAN
            elif answer.isdigit():
                processed_sample["answer_type"] = AnswerType.NUMBER
            else:
                processed_sample["answer_type"] = AnswerType.ANY

        return processed_sample

    def build_dataset(self):
         return load_dataset(self.dataset_path, data_files=self.data_files, split=self.dataset_split)