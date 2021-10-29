from fastapi import APIRouter, Request
from starlette.status import HTTP_200_OK

from .models import FilteredContent

router = APIRouter(prefix='/api')


@router.get("/filter-twitter-content", status_code=HTTP_200_OK, tags=["Twitter Content Filter"],
            response_model=FilteredContent)
def filter_twitter_content(request: Request) -> FilteredContent:
    data = await request.json()
    return FilteredContent(html_content=data.get('content'))
