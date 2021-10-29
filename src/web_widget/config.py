import os
from functools import lru_cache
from pathlib import Path

from pydantic import BaseSettings

# Internal SDKs
from datacose.secrets import Secrets
from datacose.automation import Automation


for parent in (file_path := Path().resolve()).parents:
    if (google_credentials := parent / 'service-account.json').is_file():
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = str(google_credentials.resolve())
        break

config_file = Path(__file__).resolve().parents[1] / 'config.json'
SECRETS = Secrets(client_name=Automation.initialize_client_from_config_file(config_file))


class Config(BaseSettings):
    API_KEY: str = SECRETS.get('MINDBODY_API_KEY')
    SITE_ID: str = SECRETS.get('MINDBODY_SITE_ID')
    STAFF_USERNAME: str = SECRETS.get('MINDBODY_STAFF_USERNAME')
    STAFF_PASSWORD: str = SECRETS.get('MINDBODY_STAFF_PASSWORD')
    USER_AGENT: str = SECRETS.get('MINDBODY_USER_AGENT')


@lru_cache()
def get_config():
    return Config()
