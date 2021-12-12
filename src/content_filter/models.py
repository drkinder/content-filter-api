from typing import Optional

from fastapi_utils.api_model import APIModel


class FilteredContent(APIModel):
    filter: bool
    confidence_postive: Optional[float] = None
