from fastapi import APIRouter
from starlette.status import HTTP_200_OK

from .models import FilteredContent

router = APIRouter(prefix='/content-filter')


@router.get("/filter-twitter-content", status_code=HTTP_200_OK, tags=["Twitter Content Filter"],
            response_model=FilteredContent)
def filter_twitter_content(twitter_content: str, filter_threshold: int) -> FilteredContent:
    return FilteredContent(html_content=twitter_content)
