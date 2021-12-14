import pickle
from pathlib import Path
from typing import List, Set

from gensim.parsing.preprocessing import preprocess_string, STOPWORDS
from langdetect import detect


CUSTOM_STOP_WORDS: List[str] = ['www', 'twitpic', 'tinyurl', 'com', 'https', 'http', '&amp', 'rt', 'bit', 'ly', 'bitly']
FULL_STOP: Set[str] = STOPWORDS.union(set(CUSTOM_STOP_WORDS))


def is_english(text: str) -> bool:
    """Determines if the text is in English or not.

    :param text: the text to determine if English
    :return: bool is the text English?
    """
    if not isinstance(text, str):
        return False
    return detect(text) == 'en'


def does_tweet_contain_filter_words(tweet_payload: dict) -> bool:
    """Checks whether a single Tweet's text contains any of the filtered words after applying stemming.

    :param tweet_payload: the payload of the POST request, requires "body": str and "filter_words": List[str]
    :return: bool are one of the filtered words inside of the Tweet text?
    """
    tweet_tokens: List[str] = preprocess_string(tweet_payload.get('body', ''))
    filter_tokens: List[str] = []
    for filter_word in tweet_payload.get('filter_words', []):
        if isinstance(filter_word, str):
            filter_tokens.extend(preprocess_string(filter_word))

    return any([word in tweet_tokens for word in filter_tokens])


def preprocess_tweet_body(path: Path, tweet_body: str) -> List[str]:
    """Converts a single Tweet text into a list of bigrams for classification.

    :param path: the path to the pickled phrase model
    :param tweet_body: the text content of a single Tweet
    :return: List[str] underscores between bigrams in single str
    """
    phrase_model = pickle.load(open(path, 'rb'))
    tweet_tokens = preprocess_string(tweet_body)
    tweet_tokens = [word for word in tweet_tokens if word not in FULL_STOP]
    return phrase_model[tweet_tokens]
