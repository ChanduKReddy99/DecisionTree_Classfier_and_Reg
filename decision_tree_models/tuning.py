def default_parameter_distributions() -> dict[str, list[int]]:
    return {
        "max_depth": list(range(3, 7)),
        "min_samples_split": list(range(2, 11)),
        "min_samples_leaf": list(range(1, 5)),
    }


def refined_parameter_grid(
    best_parameters: dict[str, int],
    radius: int = 2,
) -> dict[str, list[int]]:
    minimum_values = {
        "max_depth": 1,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
    }
    return {
        parameter: list(
            range(
                max(minimum_values[parameter], value - radius),
                value + radius + 1,
            )
        )
        for parameter, value in best_parameters.items()
    }
