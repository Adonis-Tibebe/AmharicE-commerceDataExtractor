import unittest
import pandas as pd
from src.utils import utils
import emoji

class TestUtils(unittest.TestCase):
    def test_replace_negative_circled(self):
        # U+1F150 is NEGATIVE CIRCLED LATIN CAPITAL LETTER A
        text = "Test \U0001F150"
        result = utils.replace_negative_circled(text)
        self.assertIn("A", result)

    def test_clean_data(self):
        df = pd.DataFrame({
            "Message": ["Hello", None, "World", "Hello"],
            "Media Path": [None, "path2", None, None]
        })
        cleaned = utils.clean_data(df)
        self.assertFalse(cleaned["Message"].isnull().any())
        self.assertTrue((cleaned["Media Path"] == "Not Available").any())
        self.assertEqual(cleaned["Message"].tolist().count("Hello"), 1)

    def test_remove_emojis_from_text(self):
        df = pd.DataFrame({"Message": ["Hello 😊", "World"]})
        result = utils.remove_emojis_from_text(df)
        self.assertNotIn("😊", result["Message"].iloc[0])
        self.assertEqual(result["Message"].iloc[1], "World")

    def test_fix_nbsp(self):
        text = "Hello\xa0World"
        fixed = utils.fix_nbsp(text)
        self.assertEqual(fixed, "Hello World")

    def test_extract_hashtags_from_text(self):
        df = pd.DataFrame({"Message": ["Buy now #sale #offer", "No tags here"]})
        result = utils.extract_hashtags_from_text(df)
        self.assertEqual(result["hashtags"].iloc[0], ["sale", "offer"])
        self.assertEqual(result["hashtags"].iloc[1], ["no tag"])
        self.assertNotIn("#", result["Message"].iloc[0])

    def test_normalize_data(self):
        df = pd.DataFrame({"Message": ["Hello 😊 #tag", "Another #test"]})
        normalized = utils.normalize_data(df)
        self.assertNotIn("😊", normalized["Message"].iloc[0])
        self.assertEqual(normalized["hashtags"].iloc[0], ["tag"])
        self.assertTrue(isinstance(normalized["Message"].iloc[0], str))
    def test_regex_tokenize(self):
        text = "አማርኛ 123, test!"
        tokens = utils.regex_tokenize(text)
        self.assertIn("አማርኛ", tokens)
        self.assertIn("123", tokens)
        self.assertIn("test", tokens)
        self.assertIn("!", tokens)

if __name__ == "__main__":
    unittest.main()