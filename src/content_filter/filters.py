import pickle
import random
from pathlib import Path
from typing import Union

from sklearn.pipeline import Pipeline

from .models import FilteredContent


def random_filter(threshold: Union[int, float]) -> bool:
    """Generates a random value between 0 and 1 and returns True if the value is less than or equal to the threshold
    param.

    :param threshold: cutoff value ranging from 0-1 inclusive
    :return: True if threshold is less than or equal to the randomly generated value else False
    """
    return random.random() <= threshold


def linear_svc_filter(processed_tweet_body: str, threshold: float) -> FilteredContent:
    """Classifies whether the processed_tweet_body should be shown on or filtered from a Twitter user's Timeline based
    on the provided threshold level.

    :param processed_tweet_body: string of bigrams e.g. "new_york big_apple ..."
    :param threshold: the cutoff value, anything with a positive percentage less or equal to will be filtered
    :return: FilteredContent with attributes filter and confidence_positive set
    """
    linear_svc_model_path: Path = Path(__file__).parent / 'resources/LinearSVCModel.sav'
    model: Pipeline = pickle.load(open(linear_svc_model_path, 'rb'))

    # Set default values if any error during classification
    is_filtered: bool = False
    prob: float = 0
    try:
        prob = model.predict_proba([processed_tweet_body])[0][1]  # [[% neg, % pos]]
        is_filtered = prob <= threshold
    except IndexError:
        print(f"Index error with model.predict_proba response in filter_linear_svc: {prob}")
    except Exception as e:
        print(f"An error occurred! {e}")
    return FilteredContent(filter=is_filtered, confidence_positive=prob)
