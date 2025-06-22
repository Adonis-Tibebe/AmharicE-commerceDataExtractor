import unittest
from telethon import TelegramClient
from config import config  # Import your config module

class TestTelegramConnection(unittest.IsolatedAsyncioTestCase):
    async def test_telegram_connection(self):
        creds = config.load_credentials()  # Use your config loader
        api_id = creds.get("api_id")
        api_hash = creds.get("api_hash")
        phone = creds.get("phone")
        self.assertIsNotNone(api_id)
        self.assertIsNotNone(api_hash)
        self.assertIsNotNone(phone)

        # Use your session file or login interactively
        client = TelegramClient("test_session", api_id, api_hash)
        await client.start(phone=phone)
        # Try to get your own user info as a lightweight connection test
        me = await client.get_me()
        self.assertIsNotNone(me)
        print(f"Connected as: {me.username or me.first_name}")

        await client.disconnect()

if __name__ == "__main__":
    unittest.main()