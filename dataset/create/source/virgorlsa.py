from pathlib import Path

import ray
from datasets import load_dataset
from PIL import Image

from dataset.const import AnswerType, ProblemType

from .inference_engine import AsyncDPDispathEngineForThink


@ray.remote
class AsyncDPDispathEngineForVIRGORLSA(AsyncDPDispathEngineForThink):
    def __init__(self, dataset_path = None, dataset_split = None, data_files = None):
        dataset_path = dataset_path or "json"
        dataset_split = dataset_split or "train"
        data_files = data_files or "/mnt/data-alpha-sg-02/team-camera/huggingface/datasets/vigorl/visual_search/vigorl_SA_RL.json"

        super().__init__(dataset_path, dataset_split, data_files)

    def preprocess(self, sample):
        processed_sample = {}
        processed_sample["source"] = "virgorlsa"
        processed_sample["images"] = [Image.open(Path("/mnt/data-alpha-sg-02/team-camera/huggingface/datasets/vigorl/") / url) for url in sample.pop("image_paths")]
        processed_sample["problem"] = sample.pop("problem").replace("Answer with the option's letter from the given choices directly.", "").replace("(A)", "Choices:\n(A)")

        image_num = processed_sample["problem"].count("<image>")
        if image_num > len(processed_sample["images"]):
            # remove extra <image> tokens
            parts = processed_sample["problem"].split("<image>")
            processed_sample["problem"] = "<image>".join(parts[: len(processed_sample["images"]) + 1])
        elif image_num < len(processed_sample["images"]):
            # add missing <image> tokens at the end
            processed_sample["problem"] = " <image>" * (len(processed_sample["images"]) - image_num) + "\n" + processed_sample["problem"]
        
        processed_sample["answer"] = sample.pop("answer")
        processed_sample["problem_type"] = ProblemType.GROUNDING

        processed_sample["answer_type"] = AnswerType.MULTIPLE_CHOICE
        processed_sample["answer"] = processed_sample["answer"].strip().upper()

        processed_sample["prompt_type"] = "normal"

        return processed_sample

    def build_dataset(self):
        return load_dataset(self.dataset_path, data_files=self.data_files, split=self.dataset_split)
