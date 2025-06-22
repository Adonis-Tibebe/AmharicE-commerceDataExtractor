import pandas as pd 
import numpy as np
import emoji
from typing import List, Dict
import re
import unicodedata

# mapping from U+1F150–U+1F169 → A–Z
NEG_CIRC_MAP = {
    0x1F150: "A", 0x1F151: "B", 0x1F152: "C", 0x1F153: "D",
    0x1F154: "E", 0x1F155: "F", 0x1F156: "G", 0x1F157: "H",
    0x1F158: "I", 0x1F159: "J", 0x1F15A: "K", 0x1F15B: "L",
    0x1F15C: "M", 0x1F15D: "N", 0x1F15E: "O", 0x1F15F: "P",
    0x1F160: "Q", 0x1F161: "R", 0x1F162: "S", 0x1F163: "T",
    0x1F164: "U", 0x1F165: "V", 0x1F166: "W", 0x1F167: "X",
    0x1F168: "Y", 0x1F169: "Z",
}

def replace_negative_circled(text: str) -> str:
    # translate any NEG_CIRC code‐point into its ASCII letter
    return text.translate(NEG_CIRC_MAP)


def clean_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the given dataframe by performing several operations:

    - Drops rows where 'Message' is NaN.
    - Fills NaN values in 'Media_Path' column with 'Not Available'.
    - Removes duplicate rows, keeping the first occurrence.

    Args:
        dataframe (pd.DataFrame): The input dataframe to be cleaned.

    Returns:
        pd.DataFrame: The cleaned dataframe.
    """
    # Make a copy of the dataframe to avoid modifying the original dataframe.
    dataframe_copy = dataframe.copy()

    # Drop rows where 'Message' is NaN.
    dataframe_copy.dropna(subset=["Message"], inplace=True)

    # Fill NaN values in 'Media_Path' column with 'Not Available'.
    dataframe_copy.fillna({"Media Path":"Not Available"}, inplace=True)

    # Remove duplicate rows, keeping the first occurrence.
    dataframe_copy.drop_duplicates(keep='first', inplace=True)

    # Return the cleaned dataframe.
    return dataframe_copy

def remove_emojis_from_text(
    df: pd.DataFrame,
    text_col: str = "Message"
) -> pd.DataFrame:
    """
    Uses the emoji library to strip all emojis from df[text_col].
    """
    df = df.copy()
    df[text_col] = df[text_col].apply(
        lambda s: emoji.replace_emoji(s, replace="") if isinstance(s, str) else s
    )
    return df

def fix_nbsp(text: str) -> str:
    # replace non-breaking spaces with ordinary spaces
    return text.replace('\xa0', ' ')


def extract_hashtags_from_text(df: pd.DataFrame, text_col: str = "Message") -> pd.DataFrame:

    # prepare a default column of "no tag" (so there are no NaNs)
    df['hashtags'] = [['no tag']] * len(df)

    # make a temporary boolean mask 
    mask = df[text_col].str.contains(r'#\w+')

    # for only those rows, extract all tags (dropping the “#”)
    df.loc[mask, 'hashtags'] = (
        df.loc[mask, text_col]  
        .str.findall(r'#(\w+)')
    )

    # now remove all “#tags” from your messages
    df[text_col] = (
        df[text_col]
        .str.replace(r'#\w+', '', regex=True)
        .str.strip()  # remove leading/trailing whitespace
    )

    df[text_col].apply(fix_nbsp) # fix non-breaking spaces that appear as /ax0 in the text

    return df

def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize data by removing emojis and extracting hashtags."""
    normalized_df = df.copy()

    # Remove emojis
    normalized_df = remove_emojis_from_text(normalized_df)

    # Extract hashtags
    normalized_df = extract_hashtags_from_text(normalized_df)
    normalized_df["Message"] = ( 
        normalized_df["Message"].str.replace(r"\s+", " ", regex=True)
        .str.strip()
) # Normalize whitespace in the 'Message' column
    
    # usage in your pipeline:
    normalized_df["Message"] = (
        normalized_df["Message"]
        .fillna("")
        .astype(str)
        .apply(lambda s: unicodedata.normalize("NFKC", s))
        .apply(replace_negative_circled)  # to remove negative circled characters which might create noise in modeling
)

    return normalized_df

def hf_tokenize(
    tokenizer,
    text: str,
    add_special_tokens: bool = False
) -> list[str]:
    """Tokenize with a pre-loaded HuggingFace tokenizer."""
    return tokenizer.tokenize(text or " ", add_special_tokens=add_special_tokens)


# Matches runs of Ethiopic syllables, Latin words/digits, or any standalone punctuation
TOKEN_RE = re.compile(r"[\u1200-\u137F]+|\w+|[^\w\s]", re.UNICODE)

def regex_tokenize(text: str) -> list[str]:
    """
    Split Amharic (Ethiopic), Latin/digits, and punctuation into separate tokens.
    Falls back to "" if text is None or empty.
    """
    return TOKEN_RE.findall(text or "")