import os
import pickle
from pathlib import Path
from typing import List, Optional

from fastapi import Request
from sklearn.pipeline import Pipeline

from .utilities import preprocess_tweet_body, does_tweet_contain_filter_words, is_english
from .filters import random_filter
from .models import FilteredContent


async def filter_twitter_content(request: Request) -> FilteredContent:
    return await filter_linear_svc(request)


async def filter_random(request: Request) -> FilteredContent:
    payload: dict = await request.json()
    return FilteredContent(filter=random_filter(payload.get('threshold', 0)), confidence_postive=0)


async def filter_linear_svc(request: Request) -> FilteredContent:
    payload: dict = await request.json()

    # If no Tweet body text in the request or Tweet is not English. Do not filter.
    if not (tweet_body := payload.get('body')) or not is_english(tweet_body):
        return FilteredContent(filter=False)

    # Check if any filter_words are in the Tweet, if so skip the sentiment model
    if does_tweet_contain_filter_words(payload):
        return FilteredContent(filter=True)

    # Load in model and preprocess tweet body text for prediction
    model_path: Path = Path('resources/LinearSVCModel.sav')
    model: Pipeline = pickle.load(open(model_path, 'rb'))
    processed_tweet_body: List[str] = preprocess_tweet_body(model_path, tweet_body)

    # Set default values if any error during classification
    is_filtered: bool = False
    prob: float = 0
    try:
        prob: List[List[float, float]] = model.predict_proba([processed_tweet_body])[0][1]  # [[% neg, % pos]]
        is_filtered = prob <= payload.get('threshold', 0.5)
    except IndexError:
        print(f"Index error with model.predict_proba response in filter_linear_svc: {prob}")
    except Exception as e:
        print(f"An error occurred! {e}")
    return FilteredContent(filter=is_filtered, confidence_postive=prob)


async def filter_multinomial_naive_bayes(request: Request) -> FilteredContent:
    payload: dict = await request.json()
    model: Pipeline = pickle.load(open(os.path.join('content_filter', 'resources', 'mnb72.pickle'), 'rb'))
    is_filtered: bool = False  # Default value if error
    prob: Optional[float] = None
    try:
        prob: List[List[float]] = model.predict_proba([payload.get('body', '')])[0][1]  # [[% neg, % pos]]
        is_filtered = prob <= payload.get('threshold', 0.5)
    except IndexError:
        print(f"Index error with model.predict_proba response in filter_multinomial_naive_bayes: "
              f"{model.predict_proba([payload.get('body', '')])}")
        return FilteredContent(filter=is_filtered)
    return FilteredContent(filter=is_filtered, confidence_postive=prob)

