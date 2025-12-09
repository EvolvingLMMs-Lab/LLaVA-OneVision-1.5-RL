import ray
from datasets import load_dataset

from dataset.const import AnswerType, ProblemType

from .inference_engine import AsyncDPDispathEngineForThink


@ray.remote
class AsyncDPDispathEngineForDocVQA(AsyncDPDispathEngineForThink):
    def __init__(self, dataset_path = None, dataset_split = None, data_files = None):
        dataset_path = dataset_path or "lmms-lab/DocVQA"
        dataset_split = dataset_split or "validation"
        data_files = data_files

        super().__init__(dataset_path, dataset_split, data_files)

    def preprocess(self, sample):
        processed_sample = {}
        processed_sample["source"] = "docvqa"
        processed_sample["images"] = [sample.pop("image")]

        processed_sample["problem"] = "<image>\n" + sample.pop("question")
        processed_sample["answer"] = sample.pop("answers")
        processed_sample["problem_type"] = ProblemType.OCR
        processed_sample["answer_type"] = AnswerType.OCRTEXT

        return processed_sample

    def build_dataset(self):
        docvqa_dataset = load_dataset(self.dataset_path, "DocVQA", split=self.dataset_split)

        return docvqa_dataset
