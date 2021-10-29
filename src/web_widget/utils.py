from functools import lru_cache

from .config import get_config

# Internal SDKs
from datacose.mindbody import MindBody


@lru_cache
def initialize_mindbody() -> MindBody:
    config = get_config()
    return MindBody(api_key=config.API_KEY, site_id=config.SITE_ID,
                    staff_username=config.STAFF_USERNAME, staff_password=config.STAFF_PASSWORD,
                    user_agent=config.USER_AGENT)

