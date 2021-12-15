from pathlib import Path

from fastapi import Request

from .utilities import does_text_contain_filter_words, is_tweet_body_valid, preprocess_text
from .filters import linear_svc_filter, random_filter
from .models import FilteredContent


PHRASE_MODEL_PATH: Path = Path(__file__).parent / 'resources/phrasemodel.sav'
DEFAULT_THRESHOLD: float = 0.5


async def filter_twitter_content(request: Request) -> FilteredContent:
    """Redirects to the best current model, the user can always get the most up-to-date version via this endpoint;
    however, still retains the opportunity to test legacy endpoints.

    Current best model: filter_linear_svc

    :param request: the POST request to the API
    :return: a FilteredContent object
    """
    return await filter_linear_svc(request)


async def filter_random(request: Request) -> FilteredContent:
    """Filters the provided tweet randomly by generating a random value in range(0, 1) inclusive and using the provided
    threshold value as a cutoff whether to filter the tweet or not.

    :param request: the POST request to the API.
        Expected structure: {
            "threshold": float  # probability positive cutoff for filtering range(0, 1) inclusive
        }
        Note: if "threshold" is not included in the body, will use the global value DEFAULT_THRESHOLD
    :return: FilteredContent with the boolean attribute filter set to true if the tweet should be filtered
    """
    payload: dict = await request.json()
    return FilteredContent(filter=random_filter(payload.get('threshold', DEFAULT_THRESHOLD)))


async def filter_linear_svc(request: Request) -> FilteredContent:
    """Follows a three-step pipeline for determining whether a given Tweet should be filtered or not.

    1. Validates the text of the Tweet to ensure it exists under the POST body key 'body', is a string, and is
       most likely in English.
    2. Checks the text for any filtered words (key: 'filter_words') or synonyms of any filtered words set by the user.
    3. Uses a Linear SVC model to predict the positivity of the text and if it is above the provided key 'threshold'
       (range(0, 1) inclusive), the Tweet will be determined positive enough to show to the user.

    :param request: the POST request to the API.
        Expected structure: {
            "body": str,  # the Tweet's text
            "threshold": float  # probability positive cutoff for filtering range(0, 1) inclusive
            "filter_words": List[str]  # list of manually added words/topics the user wishes to filter Tweets containing
        }
        "body" is the only required key/value
        "threshold" default if not provided will be set to the global DEFAULT_THRESHOLD value
        "filter_words" entirely optional
    :return: FilteredContent with the following attributes:
        filter: bool  # should the Tweet be filtered?
        confidence_positive: float  # if the Tweet body was valid and no filter_words were in the body, the model will
            return the predicted percentage that the Tweet body is of positive sentiment.
    """
    payload: dict = await request.json()

    if not is_tweet_body_valid(tweet_body := payload.get('body')):
        return FilteredContent(filter=False)

    if does_text_contain_filter_words(tweet_body, words_to_filter=payload.get('filter_words', [])):
        return FilteredContent(filter=True)

    processed_tweet_body: str = preprocess_text(PHRASE_MODEL_PATH, tweet_body)
    return linear_svc_filter(processed_tweet_body, payload.get('threshold', DEFAULT_THRESHOLD))
