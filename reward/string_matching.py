from .utils import extract_boxed_content


def string_matching_reward_fn(completions: str, answer: str):
    answers = extract_boxed_content(completions)

    if answers:
        answer_str = answers[-1]
    else:
        answer_str = ""

    if answer_str == "":
        return 0
    if answer is None:
        return 0

    if answer_str.strip().lower() == answer.strip().lower():
        return 1.0
    else:
        return 0.0