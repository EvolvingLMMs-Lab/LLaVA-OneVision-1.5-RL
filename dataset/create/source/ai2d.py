import ray
from datasets import load_dataset

from dataset.const import AnswerType, ProblemType

from .inference_engine import AsyncDPDispathEngineForThink


@ray.remote
class AsyncDPDispathEngineForAI2D(AsyncDPDispathEngineForThink):
    def __init__(self, dataset_path = None, dataset_split = None, data_files = None):
        dataset_path = dataset_path or "HuggingFaceM4/the_cauldron"
        dataset_split = dataset_split or "train"

        super().__init__(dataset_path, dataset_split, data_files)

    def preprocess(self, sample):
        processed_sample = {}
        processed_sample["source"] = "ai2d"
        processed_sample["images"] = sample["images"]

        chat = sample["texts"][0]
        question = chat["user"].replace("Answer with the letter.", "").strip()
        answer = chat["assistant"].replace("Answer: ", "").strip()

        processed_sample["problem"] = "<image>\n" + question
        processed_sample["answer"] = answer
        processed_sample["problem_type"] = ProblemType.STEM
        uppercase_letters = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        if len(answer) == 1 and answer in uppercase_letters:
            processed_sample["answer_type"] = AnswerType.MULTIPLE_CHOICE
        else:
            processed_sample["answer_type"] = AnswerType.ANY
        processed_sample["prompt_type"] = "normal"

        return processed_sample

    def build_dataset(self):
        return load_dataset(self.dataset_path, "ai2d", split=self.dataset_split)
