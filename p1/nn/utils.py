def normalize(value: float, min_: float, max_: float) -> float:
    return (value - min_) / (max_ - min_)


def denormalize(value: float, min_: float, max_: float) -> float:
    return (value + min_) * (max_ - min_)
