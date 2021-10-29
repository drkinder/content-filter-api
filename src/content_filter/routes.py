from fastapi import APIRouter, Request
from starlette.status import HTTP_200_OK

from .models import FilteredContent


async def filter_twitter_content(request: Request) -> FilteredContent:
    data = await request.json()
    return FilteredContent(html_content=data.get('content'))
