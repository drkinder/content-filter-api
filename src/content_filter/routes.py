import os
import pickle

from fastapi import Request
from typing import List
from sklearn.pipeline import Pipeline

try:
    from .filters import random_filter
    from .models import FilteredContent
except ImportError:
    from filters import random_filter
    from models import FilteredContent


async def filter_twitter_content(request: Request) -> FilteredContent:
    data: dict = await request.json()
    twitter_body: str = data.get('body')
    return FilteredContent(filter=random_filter(data.get('threshold', 0)), confidence_negative=0)


async def filter_linear_svc(request: Request) -> FilteredContent:
    data: dict = await request.json()
    # model: Pipeline = pickle.load(open(os.path.join('resources', 'LinearSVCModel79%.sav'), 'rb'))  # Local
    model: Pipeline = pickle.load(open(os.path.join('content_filter', 'resources', 'LinearSVCModel79%.sav'), 'rb'))  # Production
    is_filtered: bool = False  # Default value if problem
    try:
        prob: List[List[float, float]] = model.predict_proba([data.get('body', '')])[0][1]  # [[% negative, % positive]]
        is_filtered = prob >= data.get('threshold', 0.5)
    except IndexError:
        print(f"Index error with model.predict_proba response in filter_linear_svc: {prob}")
    return FilteredContent(filter=is_filtered, confidence_negative=prob)


async def filter_multinomial_naive_bayes(request: Request) -> FilteredContent:
    data: dict = await request.json()
    # model: Pipeline = pickle.load(open(os.path.join('resources', 'mnb72.pickle'), 'rb'))  # Local
    model: Pipeline = pickle.load(open(os.path.join('content_filter', 'resources', 'mnb72.pickle'), 'rb'))  # Production
    is_filtered: bool = False  # Default value if problem
    try:
        prob: List[List[float]] = model.predict_proba([data.get('body', '')])[0][1]  # [[% negative, % positive]]
        is_filtered = prob >= data.get('threshold', 0.5)
    except IndexError:
        print(f"Index error with model.predict_proba response in filter_multinomial_naive_bayes: {prob}")
    return FilteredContent(filter=is_filtered, confidence_negative=prob)


if __name__ == '__main__':
    data: dict = {'body': 'I hate you', 'threshold': 0.5}
    # model = pickle.load(open(os.path.join('resources', 'mnb72.pickle'), 'rb'))
    # is_filtered: bool = False  # Default value if problem
    # try:
    #     prob: List[List[float]] = model.predict_proba([data.get('body', '')])  # [[% negative, % positive]]
    #     is_filtered = prob[0][1] <= data.get('threshold', 0.5)
    # except IndexError:
    #     print(f"Index error with model.predict_proba response in filter_multinomial_naive_bayes: {prob}")
    # content = FilteredContent(filter=is_filtered)
    #
    # print(content.filter)
