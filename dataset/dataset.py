import io
import math
import re
from typing import Any, Dict, Optional, Union

from areal.utils import logging
from datasets import concatenate_datasets, load_dataset
from datasets.distributed import split_dataset_by_node
from PIL import Image
from PIL.Image import Image as ImageObject

from .preprocess import get_preprocess_func
from .prompt import PROBLEM_TYPE_SPECIAL_PROMPT

logger = logging.getLogger("Dataset")

def convert_image(
    image: Union[Dict[str, Any], ImageObject, str],
    max_pixels: Optional[int],
) -> ImageObject:
    if isinstance(image, dict):
        image = image["image"]
    elif isinstance(image, str):
        image_format = [".png", ".jpg", ".jpeg", ".bmp", ".gif"]
        image_format.extend([fmt.upper() for fmt in image_format])
        if image.endswith(tuple(image_format)):
            with Image.open(image) as img:
                image = img.copy()
        else:
            image = Image.open(io.BytesIO(image))
    elif isinstance(image, bytes):
        image = Image.open(io.BytesIO(image))

    if image.mode in ("CMYK", "YCbCr", "LAB", "HSV", "P"):
        image = image.convert("RGB")

    if max_pixels is not None and (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(
            image.height * resize_factor
        )
        image = image.resize((width, height))
    return image


def process_dataset(
    path: str,
    split: str,
    processor,
    max_length: Optional[int] = None,
    ignore_prompt_type: bool = False,
):

    # detect pattern like "[10%]" in the path and remove it, storing the percentage
    percent_match = re.search(r"\[(\d{1,3})%\]", path)
    percent = None
    if percent_match:
        percent = int(percent_match.group(1))
        path = re.sub(r"\[\d{1,3}%\]", "", path)

    if path.endswith(".parquet"):
        dataset = load_dataset("parquet", data_files=path)['train']
    elif path.endswith(".json"):
        dataset = load_dataset("json", data_files=path)['train']
    else:
        dataset = load_dataset(path=path, split=split)

    def general_process(sample):
        processed_images = [
            convert_image(image, 512 * 512) for image in sample["images"]
        ]
        if "qwen" in processor.image_processor.image_processor_type.lower():
            image_token = "<|vision_start|><|image_pad|><|vision_end|>"
        else:
            image_token = processor.image_token if processor is not None else "<image>"
        system_prompt = {
            "role": "system",
            "content": "You are a helpful AI assistant.",
        }

        thinking_prompt = (
            "Think and solve the following question step by step. "
            "Please put your thinking and analysis procedure within <think></think>. "
            "Put ONLY your final answer within <answer></answer>."
        )
        normal_prompt = "Put ONLY your final answer within <answer></answer>."

        if ignore_prompt_type:
            prompt = thinking_prompt
        else:
            if "prompt_type" in sample and sample["prompt_type"] == "normal":
                prompt = normal_prompt
            else:
                prompt = thinking_prompt

        prompt = (
            prompt
            + (f" {sample['format_guidance']}\n" if "format_guidance" in sample else "")
        )

        if str(sample["problem_type"]) in PROBLEM_TYPE_SPECIAL_PROMPT:
            prompt += PROBLEM_TYPE_SPECIAL_PROMPT[str(sample["problem_type"])]
        else:
            prompt += "\n"

        messages = [
            {
                "role": "user",
                "content": prompt + sample["problem"]
                .replace("<image>", image_token)
            }
        ]
        messages.insert(0, system_prompt)
        messages = processor.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        sample.update({"messages": messages, "images": processed_images})
        return sample

    dataset = dataset.map(
        lambda sample: general_process(get_preprocess_func(path)(sample)),
        num_proc=16,
        load_from_cache_file=True,
    )

    # Filter out sequences longer than max_length if max_length is provided
    if max_length is not None:

        def filter_length(sample):
            # Process the sample to get the total token count including image tokens
            processed_input = processor(
                text=[sample["messages"]],
                images=sample["images"] if len(sample["images"]) > 0 else None,
                padding=False,
                return_tensors="pt",
                return_length=True,
                return_attention_mask=False,
            )
            total_tokens = len(processed_input["input_ids"].squeeze(0))
            return total_tokens <= max_length

        dataset = dataset.filter(filter_length)

    if percent is not None:
        try:
            total_rows = len(dataset)
        except Exception:
            total_rows = getattr(dataset, "num_rows", None)

        if total_rows is None:
            return dataset

        take_n = max(1, math.floor(total_rows * percent / 100))
        dataset = dataset.shuffle(seed=42)
        dataset = dataset.select(range(take_n))

    return dataset


def get_dataset(
    path: str,
    split: str,
    processor,
    rank: int,
    world_size: int,
    max_length: Optional[int] = None,
    ignore_prompt_type: bool = False,
):
    if ignore_prompt_type:
        logger.info("Ignoring prompt type for thinking dataset processing.")
    path = path.split(",")
    dataset_list = [
        process_dataset(p, split, processor, max_length, ignore_prompt_type) for p in path
    ]
    dataset = concatenate_datasets(dataset_list)
    dataset = dataset.shuffle(seed=42)
    dataset = split_dataset_by_node(dataset, rank=rank, world_size=world_size)
    return dataset
