"""Client: Simple App CRUD example.

This example demonstrates how to use the dbgpt client to get, list apps.
Example:
    .. code-block:: python

        DBGPT_API_KEY = "dbgpt"
        client = Client(api_key=DBGPT_API_KEY)
        # 1. List all apps
        res = await list_app(client)
        # 2. Get an app
        res = await get_app(client, app_id="bf1c7561-13fc-4fe0-bf5d-c22e724766a8")
"""

import asyncio

from dbgpt_client import Client
from dbgpt_client.app import list_app


async def main():
    # initialize client
    DBGPT_API_KEY = "dbgpt"
    client = Client(api_key=DBGPT_API_KEY)
    try:
        res = await list_app(client)
        print(res)
    finally:
        # explicitly close client to avoid event loop closed error
        await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
