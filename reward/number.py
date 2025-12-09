import re

from .utils import extract_boxed_content


def number_reward_fn(completions: str, answer: str):
    answers = extract_boxed_content(completions)

    if answers:
        answer_str = answers[-1]
    else:
        answer_str = ""
    match = re.findall(r"([0-9\.]+)", answer_str)
    if match:
        count = match[-1]
    else:
        count = ""

    if count is None:
        return 0
    if answer is None:
        return 0
    return float(count.strip() == answer.strip())
