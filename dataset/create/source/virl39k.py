import ray
from datasets import load_dataset
from PIL import Image

from dataset.const import AnswerType, ProblemType

from .inference_engine import AsyncDPDispathEngineForThink


@ray.remote
class AsyncDPDispathEngineForViRL39K(AsyncDPDispathEngineForThink):
    def __init__(self, dataset_path = None, dataset_split = None, data_files = None):
        dataset_path = dataset_path or "parquet"
        dataset_split = dataset_split or "train"
        data_files = data_files or "./data/ViRL39K/39Krelease.parquet"
        super().__init__(dataset_path, dataset_split, data_files)

    def preprocess(self, sample):
        processed_sample = {}
        processed_sample["source"] = "virl39k"
        processed_sample["images"] = [Image.open(f"./data/ViRL39K/{url}") for url in sample.pop("image")]
        processed_sample["problem"] = sample.pop("question")

        image_num = processed_sample["problem"].count("<image>")
        if image_num > len(processed_sample["images"]):
            # remove extra <image> tokens
            parts = processed_sample["problem"].split("<image>")
            processed_sample["problem"] = "<image>".join(parts[: len(processed_sample["images"]) + 1])
        elif image_num < len(processed_sample["images"]):
            # add missing <image> tokens at the end
            processed_sample["problem"] = " <image>" * (len(processed_sample["images"]) - image_num) + "\n" + processed_sample["problem"]
        
        processed_sample["answer"] = sample.pop("answer")
        if processed_sample["answer"].startswith("\\boxed{"):
            processed_sample["answer"] = processed_sample["answer"][7:-1]

        if sample["category"] == "Spatial Reasoning":
            processed_sample["problem_type"] = ProblemType.SPATIAL_REASONING
        elif sample["category"] == "Tables/Diagrams/Charts":
            processed_sample["problem_type"] = ProblemType.OCR
        else:
            processed_sample["problem_type"] = ProblemType.STEM

        ans = processed_sample["answer"].strip()
        if len(ans) == 1 and ans.upper() in {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J"}:
            processed_sample["answer_type"] = AnswerType.MULTIPLE_CHOICE
            processed_sample["answer"] = ans.upper()
        else:
            processed_sample["answer_type"] = AnswerType.MATH_EXPRESSIONS

        return processed_sample

    def build_dataset(self):
        dataset = load_dataset(self.dataset_path, data_files=self.data_files, split=self.dataset_split)
        return dataset
