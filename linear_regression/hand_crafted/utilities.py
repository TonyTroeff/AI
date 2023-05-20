from typing import Callable

import torch


class CostParameters:
    def __init__(self, expected: torch.Tensor, predicted: torch.Tensor, weights: list[torch.Tensor], biases: list[torch.Tensor]):
        self.expected = expected
        self.predicted = predicted
        self.weights = weights
        self.biases = biases


def least_square_cost(params: CostParameters) -> torch.Tensor:
    return torch.sum(torch.pow(torch.sub(params.expected, params.predicted), 2))


def compute_ridge_regression(params: CostParameters, alpha: float) -> torch.Tensor:
    result = torch.tensor(0.)
    for w in params.weights:
        result += alpha * torch.linalg.matrix_norm(w, ord=2)
    return result


# This is used to avoid over-fitting by penalizing complex models
def ridge_cost(original_cost_func: Callable[[CostParameters], torch.Tensor], alpha: float) -> Callable[[CostParameters], torch.Tensor]:
    return lambda params: original_cost_func(params) + compute_ridge_regression(params, alpha)
