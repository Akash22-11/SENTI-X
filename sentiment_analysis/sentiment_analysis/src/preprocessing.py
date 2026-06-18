

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def download_nltk_resources():
    resources = [
        "punkt",
        "punkt_tab",
        "stopwords",
        "wordnet",
        "omw-1.4",
    ]
    for resource in resources:
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(resource, quiet=True)


class TextPreprocessor:
    """Cleans and normalizes raw text for sentiment classification."""

    def __init__(self, remove_stopwords=True, lemmatize=True):
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.stop_words = set(stopwords.words("english")) if remove_stopwords else set()
        self.lemmatizer = WordNetLemmatizer() if lemmatize else None

    def clean_text(self, text: str) -> str:
        """Lowercase, remove URLs, mentions, punctuation, and digits."""
        text = str(text).lower()
        text = re.sub(r"http\S+|www\S+|https\S+", "", text)
        text = re.sub(r"@\w+|#", "", text)
        text = re.sub(r"\d+", "", text)
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize_and_filter(self, text: str) -> list:
        """Tokenize text and optionally remove stopwords / lemmatize."""
        tokens = word_tokenize(text)

        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stop_words]

        if self.lemmatize:
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens]

        # Drop very short tokens
        tokens = [t for t in tokens if len(t) > 1]
        return tokens

    def preprocess(self, text: str) -> str:
        """Full pipeline: clean -> tokenize -> filter -> rejoin."""
        cleaned = self.clean_text(text)
        tokens = self.tokenize_and_filter(cleaned)
        return " ".join(tokens)

    def preprocess_batch(self, texts) -> list:
        """Preprocess a list/series of texts."""
        return [self.preprocess(t) for t in texts]


if __name__ == "__main__":
    download_nltk_resources()
    sample = "I LOVE this product!! Visit http://example.com #amazing @brand"
    pre = TextPreprocessor()
    print("Original :", sample)
    print("Processed:", pre.preprocess(sample))
