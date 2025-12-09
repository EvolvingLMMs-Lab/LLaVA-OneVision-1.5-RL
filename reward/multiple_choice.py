from .utils import extract_boxed_content


def multiplechoice_reward_fn(completions: str, answer: str):
    answers = extract_boxed_content(completions)

    if not answers:
        return 0

    predicted_answer = answers[-1]
    if predicted_answer == "":
        return 0
    if answer is None:
        return 0

    # Extract the choice letter from predicted_answer
    predicted_answer = predicted_answer.strip()
    # Remove leading/trailing punctuation and whitespace
    predicted_answer = predicted_answer.strip('.()')
    # Get the first character which should be the letter
    predicted_answer = predicted_answer[0].upper() if predicted_answer else ""

    return 1 if predicted_answer == answer.upper() else 0