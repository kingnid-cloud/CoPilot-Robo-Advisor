# src/utils/curves.py
from typing import List
import bisect

def piecewise_score(x: float, breakpoints: List[float], scores: List[float]) -> float:
    """
    Map x to a score using piecewise-linear segments defined by breakpoints and scores.
    breakpoints: increasing list of x positions [b0, b1, ..., bn]
    scores: list of same length giving score at each breakpoint [s0, s1, ..., sn]
    Returns interpolated score.
    """
    if not breakpoints or not scores or len(breakpoints) != len(scores):
        raise ValueError("breakpoints and scores must be same non-empty length")
    if x <= breakpoints[0]:
        return float(scores[0])
    if x >= breakpoints[-1]:
        return float(scores[-1])
    i = bisect.bisect_right(breakpoints, x) - 1
    x0, x1 = breakpoints[i], breakpoints[i+1]
    s0, s1 = scores[i], scores[i+1]
    t = (x - x0) / (x1 - x0) if x1 != x0 else 0.0
    return float(s0 + t * (s1 - s0))
