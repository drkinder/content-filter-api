from pathlib import Path

from .filters import linear_svc_filter
from .models import FilteredContent
from .utilities import preprocess_text


PATH_TO_PHRASEMODEL = Path(__file__).parent / 'resources/phrasemodel.sav'


def test_linear_svc_filter_with_positive_text():
    example_text: str = 'This is a positive text because I really love unit testing!'
    processed_text: str = preprocess_text(PATH_TO_PHRASEMODEL, example_text)
    content: FilteredContent = linear_svc_filter(processed_text, threshold=0.5)
    assert not content.filter
    assert content.confidence_positive > 0.5


def test_linear_svc_filter_with_negative_text():
    example_text: str = 'This is a negative text because I really hate unit testing!'
    processed_text: str = preprocess_text(PATH_TO_PHRASEMODEL, example_text)
    content: FilteredContent = linear_svc_filter(processed_text, threshold=0.5)
    assert content.filter
    assert content.confidence_positive < 0.5
