import base64
import io

import ray
from datasets import load_dataset
from PIL import Image

from dataset.const import AnswerType, ProblemType

from .inference_engine import AsyncDPDispathEngineForThink


@ray.remote
class AsyncDPDispathEngineForTreeBench(AsyncDPDispathEngineForThink):
    def __init__(self, dataset_path = None, dataset_split = None, data_files = None):
        dataset_path = dataset_path or "HaochenWang/TreeBench"
        dataset_split = dataset_split or "train"
        data_files = data_files

        super().__init__(dataset_path, dataset_split, data_files)

    def preprocess(self, sample):
        processed_sample = {}
        processed_sample["source"] = "treebench"
        processed_sample["images"] = [Image.open(io.BytesIO(base64.b64decode(sample.pop("image"))))]
        processed_sample["problem"] = "<image>\n" + sample["question"] + "\n" + sample["multi-choice options"]
        processed_sample["answer"] = sample.pop("answer")
        processed_sample["problem_type"] = ProblemType.GROUNDING
        processed_sample["answer_type"] = AnswerType.MULTIPLE_CHOICE

        return processed_sample

    def build_dataset(self):
        return load_dataset(self.dataset_path, split=self.dataset_split)
