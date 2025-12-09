import os

import ray
from datasets import load_dataset
from PIL import Image

from dataset.const import AnswerType, ProblemType

from .inference_engine import AsyncDPDispathEngineForThink

IMAGE_BASE_PATH = "/mnt/data-alpha-sg-02/team-camera/huggingface/datasets--lili24--UniSVG/snapshots/c84b9d3495ad8adb1ac1b508f4d0ad52289bc813/png/"

@ray.remote
class AsyncDPDispathEngineForUniSVG(AsyncDPDispathEngineForThink):
    def __init__(self, dataset_path = None, dataset_split = None, data_files = None):
        dataset_path = dataset_path or "lili24/UniSVG"
        dataset_split = dataset_split or "train"
        data_files = data_files

        super().__init__(dataset_path, dataset_split, data_files)

        self.dataset_path = dataset_path
        self.dataset_split = dataset_split
        self.data_files = data_files

    def preprocess(self, sample):
        processed_sample = {}
        processed_sample["source"] = "unisvg"
        
        # Handle image_path - use get() to avoid KeyError if missing
        image_path = sample.get("image_path")
        if image_path:
            processed_sample["images"] = [Image.open(os.path.join(IMAGE_BASE_PATH, image_path)).convert("RGB")]
        else:
            processed_sample["images"] = []
        
        processed_sample["problem"] = sample.get("q_text", "")
        processed_sample["answer"] = sample.get("a_text", "")
        processed_sample["problem_type"] = ProblemType.CODING
        processed_sample["answer_type"] = AnswerType.SVG_CODE
        processed_sample["prompt_type"] = "default"

        return processed_sample


    def build_dataset(self):
        if self.data_files:
            dataset = load_dataset(
                self.dataset_path,
                data_files=self.data_files,
                split=self.dataset_split,
            )
        else:
            dataset = load_dataset(
                self.dataset_path,
                split=self.dataset_split,
            )
        dataset = dataset.filter(
            lambda sample: sample.get("type") in ["ISVGEN"],
            num_proc=16,
        )

        return dataset