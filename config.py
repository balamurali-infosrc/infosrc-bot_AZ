# config.py
import os
from dotenv import load_dotenv

load_dotenv()


class DefaultConfig:
    """Bot Configuration"""

    PORT = 8000
    APP_ID = os.environ.get("MICROSOFT_APP_ID", "")
    APP_PASSWORD = os.environ.get("MICROSOFT_APP_PASSWORD", "")
    APP_TENANTID = os.environ.get("MICROSOFT_TENANT_ID", "")
    APP_TYPE = os.environ.get("MicrosoftAppType", "SingleTenant")

    print(f"Config Loaded: APP_ID={APP_ID}, APP_TYPE={APP_TYPE}, PORT={PORT}, TenantID={APP_TENANTID}")