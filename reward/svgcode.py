import os
import re
import tempfile
from typing import Optional

import cairosvg
import numpy as np
from areal.utils import logging
from PIL import Image
from skimage.metrics import structural_similarity as ssim

from .utils import extract_boxed_content


def fix_svg(svg_content: str) -> str:
    """Fix incomplete SVG tags."""
    if not svg_content.strip():
        return svg_content
    
    # Remove incomplete tags at the end
    svg_content = re.sub(r'<[^>]*$', '', svg_content)

    # Count opening and closing tags
    open_tags = re.findall(r'<([a-zA-Z]+)[^>]*>', svg_content)
    close_tags = re.findall(r'</([a-zA-Z]+)>', svg_content)

    # Count tag occurrences
    tag_counts = {}
    for tag in open_tags:
        tag_counts[tag] = tag_counts.get(tag, 0) + 1
    for tag in close_tags:
        tag_counts[tag] = tag_counts.get(tag, 0) - 1

    # Close unclosed tags in reverse order (stack-based approach)
    unclosed_tags = []
    for tag, count in tag_counts.items():
        if count > 0:
            unclosed_tags.extend([tag] * count)
    
    # Close tags in reverse order
    closing_tags = [f'</{tag}>' for tag in reversed(unclosed_tags)]
    svg_content += ''.join(closing_tags)

    # Ensure SVG tag is closed
    if not svg_content.strip().endswith('</svg>'):
        svg_content += '</svg>'

    return svg_content


def svg_to_png(svg_content: str, output_path: str, width: int = 336, height: int = 336, timeout: float = 5.0) -> Optional[str]:
    """Convert SVG content to PNG image.
    
    Note: Timeout is handled by ProcessPoolExecutor in AsyncRewardWrapper (15s default).
    For safety, skip image conversion if SVG is too large or complex.
    """
    try:
        # Skip if SVG content is too large (likely to cause issues)
        if len(svg_content) > 100000:  # 100KB limit
            return None
            
        # Try to fix incomplete SVG
        if not svg_content.strip().endswith('</svg>'):
            svg_content = fix_svg(svg_content)

        cairosvg.svg2png(
            bytestring=svg_content.encode('utf-8'),
            write_to=output_path,
            output_width=width,
            output_height=height,
            background_color='white'
        )
        return output_path if os.path.exists(output_path) else None
    except Exception:
        return None


def extract_svg_code(text: str) -> str:
    """Extract SVG code from text."""
    # Try to extract from <answer></answer> tags first
    boxed_matches = extract_boxed_content(text)
    if boxed_matches:
        svg_text = boxed_matches[-1].strip()
        # Check if it contains SVG
        if '<svg' in svg_text.lower():
            return svg_text

    # Look for SVG pattern
    svg_pattern = r'<svg[^>]*>.*?</svg>'
    matches = re.findall(svg_pattern, text, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()

    # Look for code blocks
    code_block_pattern = r'```(?:svg|xml)?\s*(.*?)```'
    matches = re.findall(code_block_pattern, text, re.DOTALL)
    if matches:
        svg_text = matches[-1].strip()
        if '<svg' in svg_text.lower():
            return svg_text

    # Last resort: return the text as is (might contain SVG)
    return text.strip()


def normalize_svg(svg: str) -> str:
    """Normalize SVG for comparison."""
    # Remove extra whitespace
    svg = re.sub(r'\s+', ' ', svg)
    
    # Normalize quotes
    svg = svg.replace('"', "'")
    
    # Remove comments
    svg = re.sub(r'<!--.*?-->', '', svg, flags=re.DOTALL)
    
    return svg.strip().lower()


def calculate_token_overlap(generated: str, reference: str) -> float:
    """Calculate token overlap similarity between two SVG strings."""
    # Tokenize by splitting on whitespace and special characters
    gen_tokens = set(re.findall(r'\w+|[<>=/"]|#[0-9a-fA-F]+', generated.lower()))
    ref_tokens = set(re.findall(r'\w+|[<>=/"]|#[0-9a-fA-F]+', reference.lower()))

    if not ref_tokens:
        return 0.0

    # Calculate Jaccard similarity (intersection over union)
    intersection = len(gen_tokens & ref_tokens)
    union = len(gen_tokens | ref_tokens)

    if union == 0:
        return 0.0

    return intersection / union


def calculate_structure_similarity(generated: str, reference: str) -> float:
    """Calculate structural similarity by comparing SVG elements and attributes."""
    # Extract all SVG tags and attributes
    gen_tags = re.findall(r'<([a-zA-Z]+)[^>]*>', generated.lower())
    ref_tags = re.findall(r'<([a-zA-Z]+)[^>]*>', reference.lower())

    if not ref_tags:
        return 0.0

    # Calculate tag set similarity
    gen_tag_set = set(gen_tags)
    ref_tag_set = set(ref_tags)

    intersection = len(gen_tag_set & ref_tag_set)
    union = len(gen_tag_set | ref_tag_set)

    if union == 0:
        return 0.0

    return intersection / union


def calculate_image_similarity(img1_path: str, img2_path: str) -> Optional[float]:
    """Calculate SSIM (Structural Similarity Index) between two images."""
    try:
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')

        # Ensure same size
        if img1.size != img2.size:
            img2 = img2.resize(img1.size, Image.LANCZOS)

        # Convert to numpy arrays
        arr1 = np.array(img1).astype(np.float64)
        arr2 = np.array(img2).astype(np.float64)

        # Calculate SSIM
        # For multi-channel images, specify channel_axis
        if len(arr1.shape) == 3:
            # RGB image: use channel_axis for newer versions, multichannel for older
            try:
                # Try newer API first (channel_axis)
                ssim_value = ssim(arr1, arr2, channel_axis=2, data_range=255.0)
            except TypeError:
                # Fallback to older API (multichannel)
                ssim_value = ssim(arr1, arr2, multichannel=True, data_range=255.0)
        else:
            # Grayscale image
            ssim_value = ssim(arr1, arr2, data_range=255.0)

        return float(ssim_value)
    except Exception:
        return None


def svg_code_eval_reward_fn(completions: str, answer: str) -> float:
    """
    Evaluate SVG code reward by comparing generated SVG with reference SVG.
    
    Args:
        completions: Generated text containing SVG code
        answer: Reference SVG code
    
    Returns:
        Reward score between 0.0 and 1.0
    """
    # Extract SVG code from completion
    generated_code = extract_svg_code(completions)
    reference_code = answer.strip()

    if not generated_code or '<svg' not in generated_code.lower():
        return 0.0

    gen_normalized = normalize_svg(generated_code)
    ref_normalized = normalize_svg(reference_code)

    if gen_normalized == ref_normalized:
        return 1.0

    token_score = calculate_token_overlap(generated_code, reference_code)

    structure_score = calculate_structure_similarity(generated_code, reference_code)

    image_score = 0.0
    if len(generated_code) < 50000 and len(reference_code) < 50000:
        from multiprocessing import Process, Queue

        def compute_image_score_worker(gen_code, ref_code, result_queue):
            """Worker function to compute image score in separate process."""
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    gen_png = os.path.join(tmpdir, "generated.png")
                    gen_success = svg_to_png(gen_code, gen_png) is not None
                    ref_png = os.path.join(tmpdir, "reference.png")
                    ref_success = svg_to_png(ref_code, ref_png) is not None
                    if ref_success and gen_success:
                        img_sim = calculate_image_similarity(ref_png, gen_png)
                        if img_sim is not None:
                            result_queue.put(img_sim)
                            return
                result_queue.put(0.0)
            except Exception:
                result_queue.put(0.0)

        try:
            result_queue = Queue()
            process = Process(
                target=compute_image_score_worker,
                args=(generated_code, reference_code, result_queue)
            )
            process.start()
            process.join(timeout=10)  # Wait up to 10 seconds

            if process.is_alive():
                # Process is still running, kill it
                process.terminate()
                process.join(timeout=1)
                if process.is_alive():
                    process.kill()  # Force kill if terminate doesn't work
                image_score = 0.0
            else:
                # Process completed, get result
                if not result_queue.empty():
                    image_score = result_queue.get()
                else:
                    image_score = 0.0
        except Exception:
            image_score = 0.0

    reward = 0.5 * image_score + 0.25 * (token_score + structure_score)
    # reward = 0.5 * (token_score + structure_score)

    # Ensure score is in [0, 1]
    reward = max(0.0, min(1.0, reward))

    return reward


def svg_reward_fn(completions: str, answer: str) -> float:
    """Main reward function for SVG code evaluation."""
    return svg_code_eval_reward_fn(completions, answer)
