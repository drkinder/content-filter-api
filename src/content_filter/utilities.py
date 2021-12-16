import json
import pickle
from pathlib import Path
from typing import List, Set

from gensim.parsing.preprocessing import preprocess_string, STOPWORDS
from langdetect import detect
from typeguard import check_type


CUSTOM_STOP_WORDS: List[str] = ['www', 'twitpic', 'tinyurl', 'com', 'https', 'http', '&amp', 'rt', 'bit', 'ly', 'bitly']
FULL_STOP: Set[str] = STOPWORDS.union(set(CUSTOM_STOP_WORDS))


def is_tweet_body_valid(tweet_body: str) -> bool:
    """Checks tweet_body type and language to determine if it is valid for classification.

    :param tweet_body: input body parameter to check
    :return: bool is this tweet body input value valid for further classification?
    """
    return isinstance(tweet_body, str) and is_english(tweet_body)


def is_english(text: str) -> bool:
    """Determines if the text is in English or not.

    :param text: the text to determine if English
    :raises: TypeError if text is not a str
    :return: bool is the text English?
    """
    check_type('text', text, str)
    return detect(text) == 'en'


def does_text_contain_filter_words(text: str, words_to_filter: List[str], include_synonyms: bool = True) -> bool:
    """Checks whether a single Tweet's text contains any of the filtered words after applying stemming and optionally
    looking for synonyms of all words in words_to_filter.

    :param text: the raw text (no preprocessing) string to check for filter words
    :param words_to_filter: a list of words which if contained in text will cause this function to return True
    :param include_synonyms: bool flag whether to include synonyms of all words_to_filter or use only words_to_filter
    :raises: TypeError if text is not a str
    :raises: TypeError if filter_words is not in types List[str], no error if empty list
    :return: bool are one of the filtered words inside of the Tweet text?
    """
    if not words_to_filter:
        return False

    check_type('text', text, str)
    check_type('words_to_filter', words_to_filter, List[str])

    text_tokens: List[str] = preprocess_string(text)
    tokens_to_filter: List[str] = [preprocess_string(word)[0] for word in words_to_filter]

    if include_synonyms:
        thesaurus: dict = json.load(open(Path(__file__).parent / 'resources/thesaurus_lem.json'))
        synonym_tokens: List[str] = []
        for token in tokens_to_filter:
            synonym_tokens.extend(thesaurus.get(token, []))
        tokens_to_filter += synonym_tokens

    return any(f_token in text_tokens for f_token in tokens_to_filter)


def preprocess_text(path: Path, text: str) -> str:
    """Converts a single Tweet text into a list of bigrams for classification.

    :param path: the path to the pickled phrase model
    :param text: the text to preprocess
    :raises: FileNotFoundError if the file at the provided path doesn't exist
    :return: bigrams linked by underscores, delimited by spaces e.g. new_york big_apple
    """
    path = Path(path) if isinstance(path, str) else path
    check_type('path', path, Path)
    if not path.is_file():
        raise FileNotFoundError(f"The file at the provided path cannot be found:\n{path}")

    tweet_tokens = [word for word in preprocess_string(text) if word not in FULL_STOP]
    phrase_model = pickle.load(open(path, 'rb'))
    return ' '.join(phrase_model[tweet_tokens])
