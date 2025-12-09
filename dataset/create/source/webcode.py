import ray
from datasets import load_dataset

from dataset.const import AnswerType, ProblemType

from .inference_engine import AsyncDPDispathEngineForThink


@ray.remote
class AsyncDPDispathEngineForWebcode(AsyncDPDispathEngineForThink):
    def __init__(self, dataset_path = None, dataset_split = None, data_files = None):
        dataset_path = dataset_path or "parquet"
        dataset_split = dataset_split or "train"
        data_files = data_files or "/mnt/data-alpha-sg-01/team-camera/shared/OVRL/data/webcode2m/snapshots/f53cd4a9317364e32a6fc7d99dc50761fe54715f/data/00022.parquet"

        super().__init__(dataset_path, dataset_split, data_files)

    def preprocess(self, sample):
        processed_sample = {}
        processed_sample["source"] = "webcode2m"
        processed_sample["images"] = [sample.pop("image")]
        processed_sample["problem"] = (
            "<image>\nPlease Generate the HTML code for this webpage design."
        )
        processed_sample["answer"] = sample.pop("text")
        processed_sample["problem_type"] = ProblemType.CODING
        processed_sample["answer_type"] = AnswerType.HTML_CODE

        return processed_sample

    def build_dataset(self):
        webcode_dataset = load_dataset(
            self.dataset_path,
            data_files=self.data_files,
            split=self.dataset_split,
        )

        return webcode_dataset
