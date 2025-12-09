import ast
import re

import ray
from datasets import load_dataset

from dataset.const import AnswerType, ProblemType

from .inference_engine import AsyncDPDispathEngineForThink


def preprocess_mmmu(sample):
    processed_sample = {}
    processed_sample["source"] = "mmmu"
    processed_sample["images"] = []
    for i in range(1, 8):
        image_key = f"image_{i}"
        if image_key in sample and sample[image_key] is not None:
            processed_sample["images"].append(sample.pop(image_key))
    

    question = sample["question"]
    options = ast.literal_eval(sample["options"]) if isinstance(sample["options"], str) else sample["options"]
    if len(options) > 0:
        question += "\nOptions:\n"
        for i, option in enumerate(options):
            question += f"{chr(65 + i)}. {option}\n"
    question = re.sub(r"<image\s*\d+>", "<image>", question, flags=re.IGNORECASE)
    processed_sample["problem"] = question

    if isinstance(sample["answer"], list):
        answer = sample["answer"][0]
    elif sample["answer"].startswith("[") and sample["answer"].endswith("]"):
        answer = sample["answer"][1:-1].split(",")[0].strip()
        if answer.startswith("'") and answer.endswith("'"):
            answer = answer[1:-1]
    else:
        answer = sample["answer"]
    processed_sample["answer"] = answer

    processed_sample["problem_type"] = ProblemType.STEM
    processed_sample["answer_type"] = AnswerType.MULTIPLE_CHOICE if len(sample["options"]) > 0 else AnswerType.MATH_EXPRESSIONS

    return processed_sample


@ray.remote
class AsyncDPDispathEngineForMMMUBio(AsyncDPDispathEngineForThink):
    def __init__(self, dataset_path = None, dataset_split = None, data_files = None):
        dataset_path = dataset_path or "MMMU/MMMU"
        dataset_split = dataset_split or "validation"
        data_files = data_files

        super().__init__(dataset_path, dataset_split, data_files)

    def preprocess(self, sample):
        sample = preprocess_mmmu(sample)
        sample["source"] = "mmmu_bio"
        return sample

    def build_dataset(self):
        dataset = load_dataset(self.dataset_path, "Biology", split=self.dataset_split)
        return dataset

@ray.remote
class AsyncDPDispathEngineForMMMUChem(AsyncDPDispathEngineForThink):
    def __init__(self, dataset_path = None, dataset_split = None, data_files = None):
        dataset_path = dataset_path or "MMMU/MMMU"
        dataset_split = dataset_split or "validation"
        data_files = data_files

        super().__init__(dataset_path, dataset_split, data_files)

    def preprocess(self, sample):
        sample = preprocess_mmmu(sample)
        sample["source"] = "mmmu_chem"
        return sample

    def build_dataset(self):
        dataset = load_dataset(self.dataset_path, "Chemistry", split=self.dataset_split)
        return dataset

@ray.remote
class AsyncDPDispathEngineForMMMUPhys(AsyncDPDispathEngineForThink):
    def __init__(self, dataset_path = None, dataset_split = None, data_files = None):
        dataset_path = dataset_path or "MMMU/MMMU"
        dataset_split = dataset_split or "validation"
        data_files = data_files

        super().__init__(dataset_path, dataset_split, data_files)

    def preprocess(self, sample):
        sample = preprocess_mmmu(sample)
        sample["source"] = "mmmu_phys"
        return sample

    def build_dataset(self):
        dataset = load_dataset(self.dataset_path, "Physics", split=self.dataset_split)
        return dataset

@ray.remote
class AsyncDPDispathEngineForMMMUGeo(AsyncDPDispathEngineForThink):
    def __init__(self, dataset_path = None, dataset_split = None, data_files = None):
        dataset_path = dataset_path or "MMMU/MMMU"
        dataset_split = dataset_split or "validation"
        data_files = data_files

        super().__init__(dataset_path, dataset_split, data_files)

    def preprocess(self, sample):
        sample = preprocess_mmmu(sample)
        sample["source"] = "mmmu_geo"
        return sample

    def build_dataset(self):
        dataset = load_dataset(self.dataset_path, "Geography", split=self.dataset_split)
        return dataset

@ray.remote
class AsyncDPDispathEngineForMMMUCS(AsyncDPDispathEngineForThink):
    def __init__(self, dataset_path = None, dataset_split = None, data_files = None):
        dataset_path = dataset_path or "MMMU/MMMU"
        dataset_split = dataset_split or "validation"
        data_files = data_files

        super().__init__(dataset_path, dataset_split, data_files)

    def preprocess(self, sample):
        sample = preprocess_mmmu(sample)
        sample["source"] = "mmmu_cs"
        return sample

    def build_dataset(self):
        dataset = load_dataset(self.dataset_path, "Computer_Science", split=self.dataset_split)
        return dataset