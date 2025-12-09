import re
from math import *

from .utils import extract_boxed_content


# endregion
def format_reward_fn(completion, **kwargs):
    pattern = re.compile(r".*<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    match_result = re.fullmatch(pattern, completion)

    if match_result:
        # only allow one <think> and one </think>
        if completion.count("<think>") == 1 and completion.count("</think>") == 1:
            exracted_boxes = extract_boxed_content(completion)
            boxes_ratio = sum([(len(box) + 9) for box in exracted_boxes]) / len(completion)
            if boxes_ratio > 0.2:
                return 0.0
            else:
                return 1.0
        else:
            return 0.0
    else:
        return 0.0

def must_have_bbox_reward_fn(completion, **kwargs):
    think_blocks = re.findall(r"<think>(.*?)</think>", completion, re.DOTALL)
    if not think_blocks:
        return 0.0
    bbox_re = re.compile(r"\[\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*\]")
    for block in think_blocks:
        if bbox_re.search(block):
            return 1.0
    return 0.0