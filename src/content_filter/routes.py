import os
import pickle
from typing import List, Optional

from fastapi import Request
from sklearn.pipeline import Pipeline

from .filters import random_filter
from .models import FilteredContent


async def filter_twitter_content(request: Request) -> FilteredContent:
    return filter_linear_svc(request)


async def filter_random(request: Request) -> FilteredContent:
    payload: dict = await request.json()
    return FilteredContent(filter=random_filter(payload.get('threshold', 0)), confidence_postive=0)


async def filter_linear_svc(request: Request) -> FilteredContent:
    payload: dict = await request.json()
    model: Pipeline = pickle.load(open(os.path.join('content_filter', 'resources', 'LinearSVCModel79%.sav'), 'rb'))
    is_filtered: bool = False  # Default value if problem
    try:
        prob: List[List[float, float]] = model.predict_proba([payload.get('body', '')])[0][1]  # [[% neg, % pos]]
        is_filtered = prob <= payload.get('threshold', 0.5)
    except IndexError:
        print(f"Index error with model.predict_proba response in filter_linear_svc: {prob}")
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

