import ray
from datasets import load_dataset

from dataset.const import AnswerType, ProblemType

from .inference_engine import AsyncDPDispathEngineForThink


@ray.remote
class AsyncDPDispathEngineForPixmoCount(AsyncDPDispathEngineForThink):
    def __init__(self, dataset_path = None, dataset_split = None, data_files = None):
        dataset_path = dataset_path or "./data/pixmo_count"
        dataset_split = dataset_split or "train"
        data_files = data_files

        super().__init__(dataset_path, dataset_split, data_files)

    def preprocess(self, sample):
        processed_sample = {}
        processed_sample["images"] = sample["images"]
        processed_sample["problem"] = sample["problem"]
        processed_sample["answer"] = sample["answer"]
        processed_sample["source"] = "pixmo_count"
        processed_sample["problem_type"] = ProblemType.COUNTING
        processed_sample["answer_type"] = AnswerType.NUMBER
        processed_sample["prompt_type"] = "normal"

        return processed_sample

    def build_dataset(self):
        return load_dataset(self.dataset_path, split=self.dataset_split)
