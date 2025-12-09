import ray
from datasets import load_dataset
from PIL import Image

from dataset.const import AnswerType, ProblemType

from .inference_engine import AsyncDPDispathEngineForThink


@ray.remote
class AsyncDPDispathEngineForRefL4(AsyncDPDispathEngineForThink):
    def __init__(self, dataset_path = None, dataset_split = None, data_files = None):
        dataset_path = dataset_path or "parquet"
        dataset_split = dataset_split or "train"
        data_files = data_files or "/mnt/data-alpha-sg-02/team-camera/datasets/Ref-L4/ref-l4-val.parquet"

        super().__init__(dataset_path, dataset_split, data_files)

    def preprocess(self, sample):
        processed_sample = {}
        processed_sample["source"] = "refl4"
        processed_sample["images"] = [
            Image.open(
                "/mnt/data-alpha-sg-02/team-camera/datasets/Ref-L4/images/"
                + sample.pop("file_name")
            )
        ]
        processed_sample["problem"] = (
            "<image>\n"
            + "What is the coordinates for the following: "
            + sample.pop("caption").lower()
            + "\nThe bounding box coordinates should be in the format [x_min, y_min, x_max, y_max]. The number range is from 0 to 1."
        )
        bbox = sample.pop("bbox")
        bbox[2] += bbox[0]  # x_max = x_min + width
        bbox[3] += bbox[1]  # y_max = y_min + height
        bbox[0] /= sample["width"]  # x_min
        bbox[1] /= sample["height"]  # y_min
        bbox[2] /= sample["width"]  # x_max
        bbox[3] /= sample["height"]  # y_max
        processed_sample["answer"] = f"{bbox}"
        processed_sample["problem_type"] = ProblemType.GROUNDING
        processed_sample["answer_type"] = AnswerType.BBOX
        processed_sample["prompt_type"] = "normal"

        return processed_sample

    def build_dataset(self):
        return load_dataset(
            self.dataset_path,
            data_files=self.data_files,
            split=self.dataset_split,
        )
