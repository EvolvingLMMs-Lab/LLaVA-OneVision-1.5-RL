from .utils import extract_boxed_content


# endregion
def critic_reward_fn(completion, answer: str):
    exracted_boxes = extract_boxed_content(completion)
    if not exracted_boxes:
        return 0.0
    box = exracted_boxes[-1]
    # Check if the extracted box content matches the expected answer
    if box.strip() == answer.strip():
        return 1.0
    else:
        return 0.0