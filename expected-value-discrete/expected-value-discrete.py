import numpy as np

def expected_value_discrete(x: list, p: list) -> float:
    sum = 0
    for i, j in zip(x, p):
        sum += i * j
    return float(sum)