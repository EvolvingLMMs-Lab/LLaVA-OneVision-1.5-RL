from enum import StrEnum


class ProblemType(StrEnum):
    STEM = "stem"
    OCR = "ocr"
    GENERAL = "general"
    COUNTING = "counting"
    GROUNDING = "grounding"
    SPATIAL_REASONING = "spatial-reasoning"
    CODING = "coding"
    EMBODIED = "embodied"

class AnswerType(StrEnum):
    MULTIPLE_CHOICE = "multiple-choice"
    MATH_EXPRESSIONS = "math-expressions"
    HTML_CODE = "html-code"
    SVG_CODE = "svg-code"
    GENERAL_CODE = "general-code"
    BBOX = "bbox"
    NUMBER = "number"
    CRITIC = "critic"
    BOOLEAN = "boolean"
    OCRTEXT = "ocrtext"
    ANY = "any"