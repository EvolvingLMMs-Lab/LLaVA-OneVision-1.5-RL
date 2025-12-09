import ray
from datasets import load_dataset

from dataset.const import AnswerType, ProblemType

from .inference_engine import AsyncDPDispathEngineForThink


@ray.remote
class AsyncDPDispathEngineForInfographicVQA(AsyncDPDispathEngineForThink):
    def __init__(self, dataset_path = None, dataset_split = None, data_files = None):
        dataset_path = dataset_path or "lmms-lab/DocVQA"
        dataset_split = dataset_split or "validation"
        data_files = data_files

        super().__init__(dataset_path, dataset_split, data_files)

    def preprocess(self, sample):
        processed_sample = {}
        processed_sample["source"] = "infographicvqa"
        processed_sample["images"] = [sample.pop("image")]

        if "vidore" in self.dataset_path:
            # For the training set
            processed_sample["problem"] = "<image>\n" + sample.pop("query")
            processed_sample["answer"] = sample.pop("answer").rstrip(".")
        else:
            processed_sample["problem"] = "<image>\n" + sample.pop("question")
            processed_sample["answer"] = sample.pop("answers")


        processed_sample["problem_type"] = ProblemType.OCR
        processed_sample["answer_type"] = AnswerType.OCRTEXT
        processed_sample["prompt_type"] = "normal"

        return processed_sample

    def build_dataset(self):
        if "vidore" in self.dataset_path:
            infographicvqa_dataset = load_dataset(self.dataset_path, split=self.dataset_split)
        else:
            infographicvqa_dataset = load_dataset(self.dataset_path, "InfographicVQA", split=self.dataset_split)

        return infographicvqa_dataset
