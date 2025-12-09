import re
from typing import List


def extract_boxed_content(text: str) -> List[str]:
    """
    Extracts the content inside <answer></answer> from a given string.
    """
    pattern = r'<answer>(.*?)</answer>'
    matches = re.findall(pattern, text, re.DOTALL)
    return matches