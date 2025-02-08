import numpy as np


def find_translation_vector_with_reg(points, lines, reg_lambda=0.1):
    sum_A2 = 0.0
    sum_AB = 0.0
    sum_B2 = 0.0
    sum_Ad = 0.0
    sum_Bd = 0.0

    for (x, y), (A, B, C) in zip(points, lines):
        denom = A ** 2 + B ** 2
        if denom == 0:
            raise ValueError("Line coefficients A and B cannot both be zero.")
        d = A * x + B * y + C
        sum_A2 += (A ** 2) / denom
        sum_AB += (A * B) / denom
        sum_B2 += (B ** 2) / denom
        sum_Ad += (A * d) / denom
        sum_Bd += (B * d) / denom

    # Add regularization term (modify diagonal elements)
    matrix = np.array([
        [sum_A2 + reg_lambda, sum_AB],
        [sum_AB, sum_B2 + reg_lambda]
    ])
    rhs = np.array([-sum_Ad, -sum_Bd])

    tx, ty = np.linalg.solve(matrix, rhs)  # At this point, the matrix is guaranteed to be invertible

    return np.array([tx, ty])
