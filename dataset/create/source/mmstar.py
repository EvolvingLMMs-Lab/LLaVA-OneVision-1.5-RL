import ray
from datasets import load_dataset

from dataset.const import AnswerType, ProblemType

from .inference_engine import AsyncDPDispathEngineForThink


@ray.remote
class AsyncDPDispathEngineForMMStar(AsyncDPDispathEngineForThink):
    def __init__(self, dataset_path = None, dataset_split = None, data_files = None):
        dataset_path = dataset_path or "Lin-Chen/MMStar"
        dataset_split = dataset_split or "val"
        data_files = data_files

        super().__init__(dataset_path, dataset_split, data_files)

    def preprocess(self, sample):
        processed_sample = {}
        processed_sample["source"] = "mmstar"
        processed_sample["images"] = [sample.pop("image")]
        processed_sample["problem"] = "<image>\n" + sample["question"]
        processed_sample["answer"] = sample.pop("answer")
        processed_sample["problem_type"] = ProblemType.GENERAL
        processed_sample["answer_type"] = AnswerType.MULTIPLE_CHOICE

        return processed_sample

    def build_dataset(self):
        codeforces_dataset = load_dataset(self.dataset_path, split=self.dataset_split)

        return codeforces_dataset
