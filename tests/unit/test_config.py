import os
import unittest
from config import config

class TestConfig(unittest.TestCase):
    def setUp(self):
        # Ensure .env is loaded for the test
        from dotenv import load_dotenv
        load_dotenv(dotenv_path='../.env', override=True)

    def test_load_credentials(self):
        creds = config.load_credentials(env_path='../.env')
        self.assertIn('api_id', creds)
        self.assertIn('api_hash', creds)
        self.assertIn('phone', creds)
        # These should match the values in .env.example during CI
        self.assertEqual(creds['api_id'], os.getenv('TG_API_ID'))
        self.assertEqual(creds['api_hash'], os.getenv('TG_API_HASH'))
        self.assertEqual(creds['phone'], os.getenv('phone'))

if __name__ == '__main__':
    unittest.main()