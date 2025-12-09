import ray
from datasets import load_dataset

from dataset.const import AnswerType, ProblemType

from .inference_engine import AsyncDPDispathEngineForThink


@ray.remote
class AsyncDPDispathEngineForMathVision(AsyncDPDispathEngineForThink):
    def __init__(self, dataset_path = None, dataset_split = None, data_files = None):
        dataset_path = dataset_path or "MathLLMs/MathVision"
        dataset_split = dataset_split or "test"
        data_files = data_files

        super().__init__(dataset_path, dataset_split, data_files)

    def preprocess(self, sample):
        processed_sample = {}
        processed_sample["source"] = "mathvision"
        processed_sample["images"] = [sample.pop("decoded_image")]

        question = "<image>\n" + sample.pop("question")
        if len(sample["options"]) > 0:
            question += "\nOptions:\n"
            for i, option in enumerate(sample["options"]):
                question += f"{chr(65 + i)}. {option}\n"

        processed_sample["problem"] = question
        processed_sample["answer"] = sample.pop("answer")
        processed_sample["problem_type"] = ProblemType.STEM
        processed_sample["answer_type"] = AnswerType.MULTIPLE_CHOICE if len(sample["options"]) > 0 else AnswerType.MATH_EXPRESSIONS

        return processed_sample


    def build_dataset(self):
        return load_dataset(self.dataset_path, split=self.dataset_split)
