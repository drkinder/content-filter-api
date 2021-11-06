import random
from typing import Union


def random_filter(probability: Union[int, float]) -> bool:
    return random.random() < probability
