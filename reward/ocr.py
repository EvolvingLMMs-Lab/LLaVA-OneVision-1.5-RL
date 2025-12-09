from .utils import extract_boxed_content


def levenshtein_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


def ocr_reward_fn(completions: str, answer: str | list[str]):
    answers = extract_boxed_content(completions)

    if answers:
        answer_str = answers[-1]
    else:
        answer_str = ""

    if answer_str == "":
        return 0
    if answer is None:
        return 0

    values = []
    # Unwrap predictions if it's a nested list
    for target in answer if isinstance(answer, list) else [answer]:
        # preprocess both the answers - gt and prediction
        gt_answer = " ".join(target.strip().lower().split())
        det_answer = " ".join(answer_str.strip().lower().split())

        dist = levenshtein_distance(gt_answer, det_answer)
        length = max(len(target.upper()), len(det_answer.upper()))
        values.append(0.0 if length == 0 else float(dist) / float(length))

    reward = 1 - min(values)
    if reward < 0.5:
        reward = 0

    return reward

