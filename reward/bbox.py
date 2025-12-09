import re

from .utils import extract_boxed_content


def extract_bbox(text):
    m = re.search(r'\[\s*[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?(?:\s*,\s*[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)*\s*\]', text)
    if m:
        s = m.group(0)
        bbox = [float(x.strip()) for x in s[1:-1].split(",")]
    else:
        bbox = None
    return bbox

def bbox_reward_fn(completions: str, answer: str):
    completions = extract_boxed_content(completions)

    if completions:
        completion = completions[-1]
    else:
        completion = ""

    predicted_bbox = extract_bbox(completion)
    gt_bbox = extract_bbox(answer)

    if predicted_bbox is None or gt_bbox is None:
        return 0.0
    if len(predicted_bbox) != len(gt_bbox) or len(predicted_bbox) != 4 or len(gt_bbox) != 4:
        return 0.0

    # compute IoU
    xA = max(predicted_bbox[0], gt_bbox[0])
    yA = max(predicted_bbox[1], gt_bbox[1])
    xB = min(predicted_bbox[2], gt_bbox[2])
    yB = min(predicted_bbox[3], gt_bbox[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (predicted_bbox[2] - predicted_bbox[0]) * (predicted_bbox[3] - predicted_bbox[1])
    boxBArea = (gt_bbox[2] - gt_bbox[0]) * (
        gt_bbox[3] - gt_bbox[1]
    )
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou