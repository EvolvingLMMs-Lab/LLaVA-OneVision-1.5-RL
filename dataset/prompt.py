from .const import ProblemType

PROBLEM_TYPE_SPECIAL_PROMPT = {
    str(ProblemType.SPATIAL_REASONING): " Please first generate the coordinates (x, y, x, y) of each relevant object mentioned in the problem.\n"
}