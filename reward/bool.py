from .utils import extract_boxed_content


def bool_reward_fn(completions: str, answer: str):
    answers = extract_boxed_content(completions)

    if answers:
        answer_str = answers[-1]
    else:
        answer_str = ""

    if answer_str == "":
        return 0
    if answer is None:
        return 0

    if "yes" in answer_str.lower() or "true" in answer_str.lower():
        predicted_bool = True
    elif "no" in answer_str.lower() or "false" in answer_str.lower():
        predicted_bool = False
    else:
        return 0

    if "yes" in answer.lower() or "true" in answer.lower():
        actual_bool = True
    elif "no" in answer.lower() or "false" in answer.lower():
        actual_bool = False
    else:
        return 0
    
    return float(predicted_bool == actual_bool)