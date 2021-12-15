from pathlib import Path

import pytest

from .utilities import does_text_contain_filter_words, is_english, preprocess_text


def test_does_text_contain_filter_words_with_invalid_text_type():
    for test_type in [None, {}, [], 0, 9.99]:
        with pytest.raises(TypeError) as _:
            does_text_contain_filter_words(text=test_type, words_to_filter=['Test'])


def test_does_text_contain_filter_words_with_invalid_words_to_filter_type():
    for test_type in [None, {}, [1, 2, 3], 0, 9.99, [None]]:
        with pytest.raises(TypeError) as _:
            does_text_contain_filter_words(text='testing', words_to_filter=test_type)


def test_does_text_contain_filter_words_with_exact_text_and_filter_word_match():
    assert does_text_contain_filter_words(text='This is a test', words_to_filter=['test'])


def test_does_text_contain_filter_words_with_synonym_text_and_filter_word_match():
    assert does_text_contain_filter_words(text='This is a test using the synonym automobile', words_to_filter=['car'])


def test_does_text_contain_filter_words_with_lemmatizatized_text_and_filter_word_match():
    assert does_text_contain_filter_words(text='This is testing with pytest', words_to_filter=['test'])
    assert does_text_contain_filter_words(text='This is a test', words_to_filter=['testing'])


def test_does_text_contain_filter_words_without_text_and_filter_word_match():
    assert not does_text_contain_filter_words(text='This is a sentence about something', words_to_filter=['test'])


def test_is_english_with_invalid_text_type():
    for test_type in [None, {}, [], 0, 9.99]:
        with pytest.raises(TypeError) as _:
            does_text_contain_filter_words(text=test_type)


def test_is_english_with_english_text():
    assert is_english('This is a test sentence written in English')


def test_is_english_with_spanish_text():
    assert not is_english('Esta es una oración de prueba escrita en español')


def test_is_english_with_polish_text():
    assert not is_english('To jest zdanie po polsku')


def test_preprocess_tweet_body_with_invalid_path_type():
    for test_type in [None, {}, [1, 2, 3], 0, 9.99, [None]]:
        with pytest.raises(TypeError) as _:
            preprocess_text(path=test_type, text='testing only')


def test_preprocess_tweet_body_with_invalid_text_type():
    for test_type in [None, {}, [1, 2, 3], 0, 9.99, [None]]:
        with pytest.raises(TypeError) as _:
            preprocess_text(path=Path(__file__), text=test_type)


def test_preprocess_tweet_body_with_invalid_path_location():
    with pytest.raises(FileNotFoundError) as _:
        preprocess_text(path=Path(__file__).parent / 'testing_file_that_does_not_exist.sav', text='testing only')


def test_preprocess_tweet_body_with_valid_parameters():
    path = Path(__file__).parent / 'resources/phrasemodel.sav'
    assert preprocess_text(path, text='My friday is super open as of now') == 'fridai super open'


def test_preprocess_tweet_body_with_only_stop_words():
    path = Path(__file__).parent / 'resources/phrasemodel.sav'
    assert preprocess_text(path, text='I to you the at my on') == ''
