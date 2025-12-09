import json
import math
import os
from pathlib import Path
import sys
import numpy as np
from scipy import ndimage
from tqdm import tqdm
project_root = Path(__file__).parent.parent.parent.parent
print(project_root)
sys.path.insert(0, str(project_root))
import ray
from PIL import Image

from dataset.const import AnswerType, ProblemType

from dataset.create.source.inference_engine import AsyncDPInferenceEngineForThink
from datasets import Dataset


class RefSpatialProcessor:
    """Base class containing RefSpatial processing logic (no Ray dependencies)."""

    FULL_DATASET_CONFIG = {
        "2D": [
            {
                "filename": "choice_qa.json",
                "image_root": "2D/image",
                "depth_root": "2D/depth",
                "answer_type": AnswerType.MULTIPLE_CHOICE,
            },
            {
                "filename": "reasoning_template_qa.json",
                "image_root": "2D/image",
                "depth_root": "2D/depth",
                "answer_type": AnswerType.CRITIC,
            },
        ],
        "3D": [
            {
                "filename": "choice_qa.json",
                "image_root": "3D/image",
                "depth_root": "3D/depth",
                "answer_type": AnswerType.MULTIPLE_CHOICE,
            },
            {
                "filename": "multi_view_qa.json",
                "image_root": "3D/image_multi_view",
                "depth_root": "3D/depth_multi_view",
                "answer_type": AnswerType.MULTIPLE_CHOICE,
            },
            {
                "filename": "reasoning_template_qa.json",
                "image_root": "3D/image",
                "depth_root": "3D/depth",
                "answer_type": AnswerType.CRITIC,
            },
            {
                "filename": "vacant_qa.json",
                "image_root": "3D/image",
                "depth_root": "3D/depth",
                "answer_type": AnswerType.BBOX,
            },
            {
                "filename": "visual_choice_qa.json",
                "image_root": "3D/image_visual_choice",
                "depth_root": "3D/depth",
                "answer_type": AnswerType.MULTIPLE_CHOICE,
            },
        ],
    }

    def _filter_dataset_config(self):
        """Filter dataset configuration based on refspatial_specific parameter."""
        # If 'all' is specified, return the full configuration
        if "all" in self.refspatial_specific:
            return self.FULL_DATASET_CONFIG

        # Otherwise, filter based on the specified files
        filtered_config = {"2D": [], "3D": []}

        for dimension, configs in self.FULL_DATASET_CONFIG.items():
            for config in configs:
                filename = config["filename"]
                # Check if this file is in the refspatial_specific list
                # Support both "filename.json" and "dimension/filename.json" formats
                if (
                    filename in self.refspatial_specific
                    or f"{dimension}/{filename}" in self.refspatial_specific
                    or f"{dimension.lower()}/{filename}" in self.refspatial_specific
                ):
                    filtered_config[dimension].append(config)

        return filtered_config

    def _convert_center_points_to_bboxes(self, center_points, box_size_ratio=0.1):
        """
        Convert center point(s) to normalized bounding box(es).

        Args:
            center_points: List of (x, y) tuples representing center points (normalized 0-1)
            box_size_ratio: Size of the box as ratio of image dimensions (default 0.1 = 10%)

        Returns:
            List of bboxes in format [[xmin, ymin, xmax, ymax], ...]
        """
        if not center_points:
            return []

        half_size = box_size_ratio / 2.0
        bboxes = []

        for x_center, y_center in center_points:
            xmin = max(0.0, x_center - half_size)
            ymin = max(0.0, y_center - half_size)
            xmax = min(1.0, x_center + half_size)
            ymax = min(1.0, y_center + half_size)
            bboxes.append([xmin, ymin, xmax, ymax])

        return bboxes

    def _infer_bbox_from_depth(
        self, depth_image, center_points, width, height, depth_threshold=10
    ):
        """
        Infer bounding boxes using depth map and center points.
        Finds connected regions of similar depth around each center point.

        Args:
            depth_image: Numpy array of depth values (uint8)
            center_points: List of (x, y) tuples (normalized 0-1)
            width: Image width in pixels
            height: Image height in pixels
            depth_threshold: Maximum depth difference for connected pixels (default 10)

        Returns:
            List of bboxes in format [[xmin, ymin, xmax, ymax], ...] (normalized 0-1)
        """


        if depth_image is None or not center_points:
            return None

        bboxes = []

        for x_norm, y_norm in center_points:
            # Convert normalized coordinates to pixel coordinates
            x_pixel = int(x_norm * width)
            y_pixel = int(y_norm * height)

            # Clamp to image bounds
            x_pixel = max(0, min(width - 1, x_pixel))
            y_pixel = max(0, min(height - 1, y_pixel))

            # Get depth value at center point
            center_depth = depth_image[y_pixel, x_pixel]

            # Create binary mask of pixels within depth threshold
            depth_diff = np.abs(depth_image.astype(np.int16) - center_depth)
            similar_depth_mask = depth_diff <= depth_threshold

            # Label connected components
            labeled_array, num_features = ndimage.label(similar_depth_mask)

            # Find which component contains the center point
            center_label = labeled_array[y_pixel, x_pixel]

            if center_label == 0:
                # No connected component found, return None to trigger fallback
                return None

            # Create mask for the component containing the center point
            object_mask = labeled_array == center_label

            # Find bounding box of the connected component
            rows, cols = np.where(object_mask)

            if len(rows) == 0:
                # Empty mask, return None to trigger fallback
                return None

            # Compute bounding box in pixel coordinates
            ymin_pixel, ymax_pixel = rows.min(), rows.max()
            xmin_pixel, xmax_pixel = cols.min(), cols.max()

            # Convert to normalized coordinates
            xmin_norm = xmin_pixel / width
            ymin_norm = ymin_pixel / height
            xmax_norm = (xmax_pixel + 1) / width  # +1 to include the pixel
            ymax_norm = (ymax_pixel + 1) / height

            # Clamp to [0, 1]
            xmin_norm = max(0.0, min(1.0, xmin_norm))
            ymin_norm = max(0.0, min(1.0, ymin_norm))
            xmax_norm = max(0.0, min(1.0, xmax_norm))
            ymax_norm = max(0.0, min(1.0, ymax_norm))

            bboxes.append([xmin_norm, ymin_norm, xmax_norm, ymax_norm])

        return bboxes

    def _extract_multiple_choice_letter(self, answer_text):
        """
        Extract the letter choice from multiple choice answer.

        Args:
            answer_text: Answer string like "(B)" or "(A) explanation text"

        Returns:
            Just the letter, e.g., "A", "B", "C", "D"
        """
        import re

        # Match pattern like (A), (B), etc.
        match = re.search(r"\(([A-Z])\)", answer_text)
        if match:
            return match.group(1)

        # Fallback: return original if no match found
        return answer_text

    def _infer_answer_type(self, answer_text):
        """
        Infer the answer type from the answer text content.

        Args:
            answer_text: The answer string to analyze

        Returns:
            AnswerType enum value (BBOX, MULTIPLE_CHOICE, or CRITIC)
        """
        import re

        answer_stripped = answer_text.strip()

        # (A), (B), (C), (D)
        mc_pattern = r"^\s*\([A-Z]\)"
        if re.search(mc_pattern, answer_stripped):
            return AnswerType.MULTIPLE_CHOICE

        # [(x, y)] or [(x, y), (x, y), ...]
        list_of_tuples_pattern = r"^\s*\[\s*\([0-9.]+\s*,\s*[0-9.]+\)"
        if re.match(list_of_tuples_pattern, answer_stripped):
            return AnswerType.BBOX

        # [[x, y, x, y]] or [[x, y, x, y], ...]
        list_of_lists_pattern = (
            r"^\s*\[\s*\[\s*[0-9.]+\s*,\s*[0-9.]+\s*,\s*[0-9.]+\s*,\s*[0-9.]+"
        )
        if re.match(list_of_lists_pattern, answer_stripped):
            return AnswerType.BBOX


        return AnswerType.CRITIC

    def _load_images(self, sample, image_root):
        """
        Load images from sample and return them with dimensions.

        Args:
            sample: Sample dict containing image field
            image_root: Root directory for images

        Returns:
            Tuple of (images_list, width, height)
        """
        images = []
        width, height = None, None

        # Extract image paths from sample
        if "image" in sample:
            image_field = sample["image"]
            image_paths = [image_field] if isinstance(image_field, str) else image_field
        else:
            image_paths = []

        for img_path in image_paths:
            full_path = (
                os.path.join(image_root, img_path)
                if not os.path.isabs(img_path)
                else img_path
            )
            if os.path.exists(full_path):
                with Image.open(full_path) as img:
                    img_copy = img.copy()  # Copy to keep data after file closes
                    images.append(img_copy)
                    # Capture dimensions from first image
                    if width is None and height is None:
                        width, height = img_copy.size
            else:
                print(f"Warning: Image not found: {full_path}")

        return images, width, height

    def _load_depth_images(self, sample, depth_root):
        """
        Load depth images from sample.

        Args:
            sample: Sample dict containing depth field (or fallback to image field)
            depth_root: Root directory for depth images

        Returns:
            List of depth images (grayscale numpy arrays)
        """
        import numpy as np

        depth_images = []

        # First try to use the "depth" field if it exists
        if "depth" in sample:
            depth_field = sample["depth"]
            depth_paths = [depth_field] if isinstance(depth_field, str) else depth_field
        elif "image" in sample:
            # Fallback: derive depth paths from image paths
            image_field = sample["image"]
            image_paths = [image_field] if isinstance(image_field, str) else image_field
            # Try to replace "_image" with "_depth" in filenames
            depth_paths = [
                img_path.replace("_image.", "_depth.") for img_path in image_paths
            ]
        else:
            depth_paths = []

        for depth_path in depth_paths:
            full_path = (
                os.path.join(depth_root, depth_path)
                if not os.path.isabs(depth_path)
                else depth_path
            )
            if os.path.exists(full_path):
                # Load as grayscale (all RGB channels should be the same for depth maps)
                with Image.open(full_path) as depth_img:
                    # Convert to numpy array for processing (this copies the data)
                    depth_array = np.array(depth_img.convert("L"), dtype=np.uint8)
                    depth_images.append(depth_array)
            else:
                # Depth image not found, append None
                depth_images.append(None)

        return depth_images

    def _extract_qa_pairs(self, sample):
        """
        Extract question-answer pairs from a sample.

        Args:
            sample: Sample dict with conversations or question/answer fields

        Returns:
            List of (question, answer) tuples
        """
        qa_pairs = []

        if "conversations" in sample:
            conversations = sample["conversations"]
            questions = [
                turn["value"] for turn in conversations if turn.get("from") == "human"
            ]
            answers = [
                turn["value"] for turn in conversations if turn.get("from") == "gpt"
            ]
            qa_pairs = list(zip(questions, answers))
        else:
            # Fallback for other formats
            problem = sample.get("question", sample.get("problem", ""))
            answer = sample.get("answer", "")
            qa_pairs.append((problem, answer))

        return qa_pairs

    def _normalize_image_tokens(self, problem, num_images):
        """
        Ensure <image> tokens match the number of images.

        Args:
            problem: Problem text with <image> tokens
            num_images: Expected number of images

        Returns:
            Normalized problem text
        """
        image_num = problem.count("<image>")
        if image_num > num_images:
            # Remove extra <image> tokens
            parts = problem.split("<image>")
            problem = "<image>".join(parts[: num_images + 1])
        elif image_num < num_images:
            # Add missing <image> tokens at the beginning
            problem = " <image>" * (num_images - image_num) + "\n" + problem

        return problem

    def _format_bbox_answer(self, problem, depth_images, answer, width, height):
        """
        Format BBOX type answers with coordinate conversion.
        Uses depth map for accurate bbox inference when available.

        Args:
            problem: Original problem text
            depth_images: List of depth images (numpy arrays), may contain None values
            answer: Answer text containing coordinates
            width: Image width
            height: Image height

        Returns:
            Tuple of (formatted_problem, formatted_answer)
        """
        formatted_problem = (
            "<image>\n"
            + "Generate the bounding box coordinates for the area of the following caption: "
            + problem.split("<image>\n")[-1]
            + "\nThe bounding box coordinates should be in the format [x_min, y_min, x_max, y_max], where all values are between 0 and 1."
            + f" The width and height of the image are {width} and {height} respectively."
        )

        try:
            import ast
            import re

            # Try to extract list of tuples from the answer string
            list_pattern = r"\[\s*\([\d\s.,]+\)(?:\s*,\s*\([\d\s.,]+\))*\s*\]"
            match = re.search(list_pattern, answer)

            if match:
                center_points = ast.literal_eval(match.group(0))
            else:
                center_points = ast.literal_eval(answer)

            if isinstance(center_points, list) and center_points:
                # Try to use depth-based inference for more accurate bboxes
                depth_image = depth_images[0] if depth_images else None
                normalized_bboxes = None

                if depth_image is not None:
                    try:
                        normalized_bboxes = self._infer_bbox_from_depth(
                            depth_image, center_points, width, height
                        )
                    except Exception as e:
                        print(
                            f"Warning: Depth-based bbox inference failed: {e}. "
                            "Falling back to simple box method."
                        )

                # Fallback to simple box method if depth inference failed or unavailable
                if normalized_bboxes is None:
                    normalized_bboxes = self._convert_center_points_to_bboxes(
                        center_points
                    )

                formatted_answer = str(normalized_bboxes)
            else:
                formatted_answer = answer
        except Exception as e:
            print(f"Warning: Failed to parse BBOX answer '{answer}': {e}")
            formatted_answer = answer

        return formatted_problem, formatted_answer

    def _format_multiple_choice_answer(self, problem, answer):
        """
        Format MULTIPLE_CHOICE type answers.

        Args:
            problem: Original problem text
            answer: Answer text containing choice letter

        Returns:
            Tuple of (problem, formatted_answer)
        """
        return problem, self._extract_multiple_choice_letter(answer)

    def _format_critic_answer(self, problem, answer):
        """
        Format CRITIC type answers (no-op, returns as-is).

        Args:
            problem: Original problem text
            answer: Answer text

        Returns:
            Tuple of (problem, answer)
        """
        return problem, answer

    def _process_single_turn(
        self,
        problem,
        answer,
        images,
        depth_images,
        width,
        height,
        base_id,
        turn_idx,
        total_turns,
    ):
        """
        Process a single conversation turn into a sample.

        Args:
            problem: Question text
            answer: Answer text
            images: List of PIL images
            width: Image width
            height: Image height
            base_id: Base sample ID
            turn_idx: Turn index in conversation
            total_turns: Total number of turns

        Returns:
            Processed sample dict
        """
        processed_sample = {}

        # # Create unique ID for this turn
        # if total_turns > 1:
        #     processed_sample["id"] = f"refspatial_{base_id}_turn{turn_idx}"
        # else:
        #     processed_sample["id"] = f"refspatial_{base_id}"

        processed_sample["source"] = "refspatial"
        processed_sample["images"] = images
        processed_sample["problem_type"] = str(ProblemType.EMBODIED)

        # Normalize <image> tokens
        problem = self._normalize_image_tokens(problem, len(images))

        # Infer answer type and format accordingly
        answer_type = self._infer_answer_type(answer)
        processed_sample["answer_type"] = str(answer_type)

        if answer_type == AnswerType.BBOX:
            problem, answer = self._format_bbox_answer(
                problem, depth_images, answer, width, height
            )
        elif answer_type == AnswerType.MULTIPLE_CHOICE:
            problem, answer = self._format_multiple_choice_answer(problem, answer)
        else:  # CRITIC
            problem, answer = self._format_critic_answer(problem, answer)

        processed_sample["problem"] = problem
        processed_sample["answer"] = answer

        return processed_sample

    def _preprocess_sample(self, sample, config, base_path, dimension, json_file, idx):
        """
        Preprocess a single RefSpatial sample, handling multi-turn conversations.

        Returns:
            List of processed samples (one per conversation turn)
        """
        # Load images and depth (shared across all turns)
        image_root = os.path.join(base_path, config["image_root"])
        depth_root = os.path.join(base_path, config["depth_root"])

        images, width, height = self._load_images(sample, image_root)
        depth_images = self._load_depth_images(sample, depth_root)

        # Extract question-answer pairs
        qa_pairs = self._extract_qa_pairs(sample)

        # Generate base ID for all turns
        base_id = sample.get("id", f"{dimension}_{Path(json_file).stem}_{idx}")

        # Process each turn as a separate sample
        processed_samples = []
        for turn_idx, (problem, answer) in enumerate(qa_pairs):
            processed_sample = self._process_single_turn(
                problem=problem,
                answer=answer,
                images=images,
                depth_images=depth_images,
                width=width,
                height=height,
                base_id=base_id,
                turn_idx=turn_idx,
                total_turns=len(qa_pairs),
            )
            processed_samples.append(processed_sample)

        return processed_samples


@ray.remote
class AsyncDPInferenceEngineForRefSpatial(
    RefSpatialProcessor, AsyncDPInferenceEngineForThink
):
    """RefSpatial/
    ├── 2D/
    │   ├── choice_qa.json (Enhances basic spatial multiple-choice question abilities; large volume)
    │   └── reasoning_template_qa.json (Enhances understanding of basic spatial relationships, concepts, and spatial referring; large volume)
    ├── 3D/
    │   ├── choice_qa.json (Enhances basic spatial multiple-choice question abilities with more spatial concepts and relationships; more precise)
    │   ├── reasoning_template_qa.json (Enhances deeper understanding of spatial concepts, relationships, and spatial referring; more precise)
    │   ├── vacant_qa.json (Enhances ability to refer to vacant positions, we edit this to bounding box format for training)
    │   └── visual_choice_qa.json (Enhances visual prompt understanding in benchmarks, such as different colored boxes or point expressions; reduces the gap between benchmark and training)
    """

    def __init__(
        self,
        dataset_path=None,
        dataset_split=None,
        data_files=None,
        refspatial_specific=None,
        dataset_size=None,
        to_parquet_only=False,
        incremental=False,
        parquet_path=None,
    ):
        dataset_split = dataset_split or "train"
        self.dataset_path = dataset_path or "./data"
        self.refspatial_specific = refspatial_specific or ["all"]
        self.to_parquet_only = to_parquet_only
        self.incremental = incremental
        self.parquet_path = parquet_path
        self.existing_sample_count = 0

        # Load existing parquet to get sample count if incremental mode is enabled
        if self.incremental and self.parquet_path:
            self._load_existing_parquet_count()

        self.dataset_config = self._filter_dataset_config()
        print(f"Class init before super")
        super().__init__(dataset_path, dataset_split, data_files, dataset_size)
        print(f"Class init after super")

    def _load_existing_parquet_count(self):
        """Load existing parquet file(s) to get the count of already processed samples."""
        import pyarrow.parquet as pq
        from pathlib import Path

        if not self.parquet_path:
            print("No parquet path specified, starting from scratch")
            self.existing_sample_count = 0
            return

        parquet_path = Path(self.parquet_path)

        # Check if it's a directory (contains multiple parquet shards)
        if parquet_path.is_dir():
            try:
                parquet_files = sorted(parquet_path.glob("*.parquet"))
                total_rows = 0
                for pq_file in parquet_files:
                    parquet_file = pq.ParquetFile(str(pq_file))
                    total_rows += parquet_file.metadata.num_rows
                self.existing_sample_count = total_rows
                print(f"Found {len(parquet_files)} existing parquet files with {self.existing_sample_count} total samples, will resume from there")
            except Exception as e:
                print(f"Warning: Failed to read existing parquet directory: {e}")
                self.existing_sample_count = 0
        # Check if it's a single file
        elif parquet_path.is_file():
            try:
                parquet_file = pq.ParquetFile(str(parquet_path))
                self.existing_sample_count = parquet_file.metadata.num_rows
                print(f"Found existing parquet with {self.existing_sample_count} samples, will resume from there")
            except Exception as e:
                print(f"Warning: Failed to read existing parquet: {e}")
                self.existing_sample_count = 0
        else:
            print(f"No existing parquet found at {self.parquet_path}, starting from scratch")
            self.existing_sample_count = 0


    def build_dataset(self):
        """Build dataset from RefSpatial JSON files."""
        all_samples = []
        sample_idx = 0
        global_sample_idx = 0  # Track global position for incremental mode

        # Count total number of JSON files after filtering
        total_files = sum(len(configs) for configs in self.dataset_config.values())

        # Calculate samples per file (rounded up)
        if self.dataset_size is not None and total_files > 0:
            samples_per_file = math.ceil(self.dataset_size / total_files)
        else:
            samples_per_file = None

        for dimension, json_configs in self.dataset_config.items():
            print(f"Processing dimension: {dimension}")
            for config in json_configs:
                json_file = config["filename"]
                json_path = os.path.join(self.dataset_path, f"{dimension}", json_file)

                with open(json_path, "r") as f:
                    data = json.load(f)

                if isinstance(data, dict):
                    data = list(data.values())

                samples_from_this_file = 0

                for item in tqdm(data, desc=f"{dimension}/{json_file}"):
                    if self.dataset_size is not None and len(all_samples) + self.existing_sample_count >= self.dataset_size:
                        return Dataset.from_list(all_samples)

                    if samples_per_file is not None and samples_from_this_file >= samples_per_file:
                        break

                    try:
                        if self.to_parquet_only:
                            # Lazy mode: store metadata only, load images on demand
                            qa_pairs = self._extract_qa_pairs(item)
                            for turn_idx, (problem, answer) in enumerate(qa_pairs):
                                # Skip already processed samples in incremental mode
                                if self.incremental and global_sample_idx < self.existing_sample_count:
                                    global_sample_idx += 1
                                    samples_from_this_file += 1
                                    continue

                                # FIX: Store minimal metadata instead of full item dict to reduce memory
                                lazy_sample = {
                                    "_lazy": True,
                                    "_image_field": item.get("image"),  # Just the image paths
                                    "_depth_field": item.get("depth"),  # Just the depth paths
                                    "_item_id": item.get("id"),
                                    "_config": config,
                                    "_dimension": dimension,
                                    "_json_file": json_file,
                                    "_sample_idx": sample_idx,
                                    "_turn_idx": turn_idx,
                                    "_problem": problem,
                                    "_answer": answer,
                                    "_total_turns": len(qa_pairs),
                                }
                                all_samples.append(lazy_sample)
                                samples_from_this_file += 1
                                global_sample_idx += 1
                        else:
                            # Normal mode: load images during build
                            processed_samples = self._preprocess_sample(
                                item,
                                config,
                                self.dataset_path,
                                dimension,
                                json_file,
                                sample_idx,
                            )

                            # Skip already processed samples in incremental mode
                            for processed_sample in processed_samples:
                                if self.incremental and global_sample_idx < self.existing_sample_count:
                                    global_sample_idx += 1
                                    samples_from_this_file += 1
                                    continue
                                all_samples.append(processed_sample)
                                samples_from_this_file += 1
                                global_sample_idx += 1
                        sample_idx += 1
                    except Exception as e:
                        print(f"Error processing sample from {json_file}: {e}")
                        continue

        if self.incremental:
            print(f"Incremental mode: skipped {self.existing_sample_count} existing samples, added {len(all_samples)} new samples")

        return Dataset.from_list(all_samples)

    def fetch_task(self):
        """Override fetch_task to handle lazy loading for parquet-only mode."""
        try:
            sample = next(self.dataset_iter)

            if self.to_parquet_only and sample.get("_lazy"):
                # Load and process the sample now (using minimal metadata)
                # FIX: Reconstruct item from minimal stored fields
                item = {
                    "image": sample["_image_field"],
                    "depth": sample.get("_depth_field"),
                    "id": sample.get("_item_id"),
                }
                config = sample["_config"]
                dimension = sample["_dimension"]
                json_file = sample["_json_file"]
                sample_idx = sample["_sample_idx"]
                turn_idx = sample["_turn_idx"]
                problem = sample["_problem"]
                answer = sample["_answer"]
                total_turns = sample["_total_turns"]

                # Load images and depth
                image_root = os.path.join(self.dataset_path, config["image_root"])
                depth_root = os.path.join(self.dataset_path, config["depth_root"])

                images, width, height = self._load_images(item, image_root)
                depth_images = self._load_depth_images(item, depth_root)

                # Generate base ID
                base_id = item.get("id", f"{dimension}_{Path(json_file).stem}_{sample_idx}")

                # Process this turn
                sample = self._process_single_turn(
                    problem=problem,
                    answer=answer,
                    images=images,
                    depth_images=depth_images,
                    width=width,
                    height=height,
                    base_id=base_id,
                    turn_idx=turn_idx,
                    total_turns=total_turns,
                )

                # FIX: Cleanup depth images after processing to prevent memory leak
                del depth_images

            sample = self.preprocess(sample)
            task_id = self.next_task_id
            self.next_task_id += 1
            task = self._make_task(sample)
            task["task_id"] = task_id
            self.task_dict[task_id] = task
            return task
        except StopIteration:
            return None
        except Exception as e:
            print(f"Error in fetch_task: {e}")
            return None
