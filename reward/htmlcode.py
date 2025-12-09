import re

from .utils import extract_boxed_content


def extract_html_code(text: str) -> str:
    # Try to extract from <answer></answer> tags first
    boxed_matches = extract_boxed_content(text)
    if boxed_matches:
        return boxed_matches[-1].strip()

    # Fallback: look for HTML patterns
    # Check if there's <!DOCTYPE html> or <html> tag
    html_pattern = r'(?:<!DOCTYPE html>|<html>).*?(?:</html>|$)'
    matches = re.findall(html_pattern, text, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()

    # If still no match, try to find any substantial code block
    # Look for code wrapped in ``` or just return cleaned text
    code_block_pattern = r'```(?:html)?\s*(.*?)```'
    matches = re.findall(code_block_pattern, text, re.DOTALL)
    if matches:
        return matches[-1].strip()

    # Last resort: return the text as is (cleaned)
    return text.strip()


def normalize_html(html: str) -> str:
    # Remove extra whitespace
    html = re.sub(r'\s+', ' ', html)

    # Normalize quotes
    html = html.replace('"', "'")

    # Remove comments
    html = re.sub(r'<!--.*?-->', '', html, flags=re.DOTALL)

    return html.strip().lower()


def calculate_token_overlap(generated: str, reference: str) -> float:
    # Tokenize by splitting on whitespace and special characters
    gen_tokens = set(re.findall(r'\w+|[<>=/"]', generated.lower()))
    ref_tokens = set(re.findall(r'\w+|[<>=/"]', reference.lower()))

    if not ref_tokens:
        return 0.0

    # Calculate Jaccard similarity (intersection over union)
    intersection = len(gen_tokens & ref_tokens)
    union = len(gen_tokens | ref_tokens)

    if union == 0:
        return 0.0

    return intersection / union


def calculate_tag_structure_similarity(generated: str, reference: str) -> float:
    # Extract all HTML tags
    gen_tags = re.findall(r'<(/?\w+)', generated.lower())
    ref_tags = re.findall(r'<(/?\w+)', reference.lower())

    if not ref_tags:
        return 0.0

    # Calculate sequence similarity using longest common subsequence
    gen_tag_set = set(gen_tags)
    ref_tag_set = set(ref_tags)

    intersection = len(gen_tag_set & ref_tag_set)
    union = len(gen_tag_set | ref_tag_set)

    if union == 0:
        return 0.0

    return intersection / union


def code_eval_reward_fn(completions: str, answer: str, **kwargs) -> float:
    # Extract HTML code from completion
    generated_code = extract_html_code(completions)
    reference_code = answer.strip()

    # Normalize both for comparison
    gen_normalized = normalize_html(generated_code)
    ref_normalized = normalize_html(reference_code)

    # Check for exact match (rare but give full reward)
    if gen_normalized == ref_normalized:
        return 1.0

    # Calculate token overlap score (weight: 0.6)
    token_score = calculate_token_overlap(generated_code, reference_code)

    # Calculate tag structure similarity (weight: 0.4)
    structure_score = calculate_tag_structure_similarity(generated_code, reference_code)

    # Combined score
    reward = 0.6 * token_score + 0.4 * structure_score

    # Ensure score is in [0, 1]
    reward = max(0.0, min(1.0, reward))

    return reward


def html_reward_fn(completions: str, answer: str) -> float:
    return code_eval_reward_fn(completions, answer)
