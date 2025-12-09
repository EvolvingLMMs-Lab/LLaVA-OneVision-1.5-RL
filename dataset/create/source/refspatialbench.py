import ray
from datasets import load_dataset
import numpy as np

from dataset.const import AnswerType, ProblemType

from .inference_engine import AsyncDPInferenceEngineForThink


@ray.remote
class AsyncDPInferenceEngineForRefSpatialBench(AsyncDPInferenceEngineForThink):
    def __init__(
        self,
        dataset_path=None,
        dataset_split=None,
        data_files=None,
        dataset_size=None,
        to_parquet_only=False,
    ):
        dataset_path = dataset_path or "JingkunAn/RefSpatial-Expand-Bench"
        dataset_split = dataset_split or "placement"
        data_files = data_files
        self.to_parquet_only = to_parquet_only

        super().__init__(dataset_path, dataset_split, data_files, dataset_size)

    def _mask_to_bbox(self, mask_image):
        """
        Convert a binary mask (PIL Image) to a normalized bounding box.

        Args:
            mask_image: PIL Image with binary mask (0 or 255 values)

        Returns:
            List with single bbox in format [[xmin, ymin, xmax, ymax]] (normalized 0-1)
            Returns [[0.0, 0.0, 0.0, 0.0]] if mask is empty
        """
        # Convert PIL Image to numpy array
        mask_array = np.array(mask_image)

        # Handle different mask formats (grayscale or RGB)
        if len(mask_array.shape) == 3:
            # If RGB, take first channel (they should all be the same for binary mask)
            mask_array = mask_array[:, :, 0]

        # Find non-zero pixels (the mask)
        rows, cols = np.where(mask_array > 0)

        if len(rows) == 0:
            # Empty mask - return zero bbox
            return [[0.0, 0.0, 0.0, 0.0]]

        # Get image dimensions
        height, width = mask_array.shape

        # Compute bounding box in pixel coordinates
        ymin_pixel = rows.min()
        ymax_pixel = rows.max()
        xmin_pixel = cols.min()
        xmax_pixel = cols.max()
        # Normalize to [0, 1]
        xmin_norm = xmin_pixel / width
        ymin_norm = ymin_pixel / height
        xmax_norm = (xmax_pixel + 1) / width  # +1 to include the pixel
        ymax_norm = (ymax_pixel + 1) / height

        # Clamp to [0, 1] (should already be in range, but safety check)
        xmin_norm = max(0.0, min(1.0, xmin_norm))
        ymin_norm = max(0.0, min(1.0, ymin_norm))
        xmax_norm = max(0.0, min(1.0, xmax_norm))
        ymax_norm = max(0.0, min(1.0, ymax_norm))

        return [[xmin_norm, ymin_norm, xmax_norm, ymax_norm]]

    def preprocess(self, sample):
        processed_sample = {}
        processed_sample["source"] = "refspatialexpandbench"
        processed_sample["images"] = [sample.pop("image")]
        processed_sample["problem"] = "<image>\n" + sample["prompt"] + " " + sample["suffix"]


        # Convert binary mask to bounding box
        mask = sample["mask"]
        bbox = self._mask_to_bbox(mask)
        processed_sample["answer"] = str(bbox)

        processed_sample["problem_type"] = ProblemType.EMBODIED
        processed_sample["answer_type"] = AnswerType.BBOX

        return processed_sample

    def build_dataset(self):
        return load_dataset(self.dataset_path, split=self.dataset_split)
