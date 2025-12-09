import json

import ray
from datasets import load_dataset

from dataset.const import AnswerType, ProblemType

from .inference_engine import AsyncDPDispathEngineForThink


@ray.remote
class AsyncDPDispathEngineForCodeforces(AsyncDPDispathEngineForThink):
    def __init__(self, dataset_path = None, dataset_split = None, data_files = None):
        dataset_path = dataset_path or "open-r1/codeforces"
        dataset_split = dataset_split or "train"
        data_files = data_files

        super().__init__(dataset_path, dataset_split, data_files)

    def preprocess(self, sample):
        processed_sample = {}
        processed_sample["source"] = "codeforces"
        processed_sample["images"] = []
        processed_sample["problem"] = (
            f"Here is a programming problem:\n{sample.pop('description')}\nThe input format is: {sample.pop('input_format')} And it's provided as a string. You need to first parse it.\nThe output format is: {sample.pop('output_format')}\nPlease ONLY provide a solution function in Python like this:\n<answer>def solution(input):\n    ...\n    return output</answer>. Please don't output any other code such as test cases, print statements, or main function etc."
        )
        processed_sample["answer"] = json.dumps(
            {
                "tests": sample.pop("official_tests"),
                "checker": sample.pop("generated_checker"),
            }
        )
        processed_sample["problem_type"] = ProblemType.CODING
        processed_sample["answer_type"] = AnswerType.GENERAL_CODE

        return processed_sample

    def build_dataset(self):
        codeforces_dataset = load_dataset(self.dataset_path, "verifiable", split=self.dataset_split).filter(
            lambda x: x["generated_checker"] is None
        )

        return codeforces_dataset
