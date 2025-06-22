import unittest
from unittest.mock import AsyncMock, MagicMock, patch
from src.services import telegram_scrapper

class TestTelegramScrapper(unittest.IsolatedAsyncioTestCase):
    async def test_scrape_channel(self):
        # Mock client and its methods
        mock_client = MagicMock()
        mock_entity = MagicMock()
        mock_entity.title = "Test Channel"
        mock_client.get_entity = AsyncMock(return_value=mock_entity)
        
        # Mock messages
        mock_message = MagicMock()
        mock_message.id = 1
        mock_message.message = "Hello 😊"
        mock_message.date = "2024-01-01"
        mock_message.media = None  # No media for this test

        # Mock iter_messages to yield our mock_message
        async def mock_iter_messages(*args, **kwargs):
            yield mock_message
        mock_client.iter_messages = mock_iter_messages

        # Mock writer
        mock_writer = MagicMock()
        media_dir = "test_media_dir"

        # Patch emoji.replace_emoji to just return the string for simplicity
        with patch("src.services.telegram_scrapper.emoji.replace_emoji", lambda msg, _: msg):
            await telegram_scrapper.scrape_channel(mock_client, "test_channel", mock_writer, media_dir)

        # Assert writer.writerow was called with expected values
        mock_writer.writerow.assert_any_call(
            ["Test Channel", "test_channel", 1, "Hello 😊", "2024-01-01", None]
        )

if __name__ == "__main__":
    unittest.main()