"""
text_cleaning.py
=================
Common NLP preprocessing utilities used across all three members' pipelines:
cleaning, tokenization, stop-word removal, lemmatization, and stemming.

Run once to download required NLTK resources:
    python -m src.text_cleaning --download
"""

import re
import string
import argparse

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer

NLTK_RESOURCES = [
    "punkt",
    "punkt_tab",
    "stopwords",
    "wordnet",
    "omw-1.4",
]


def download_nltk_resources():
    for resource in NLTK_RESOURCES:
        try:
            nltk.download(resource, quiet=True)
        except Exception as e:  # pragma: no cover
            print(f"Could not download {resource}: {e}")


_lemmatizer = WordNetLemmatizer()
_stemmer = PorterStemmer()
_stop_words = None


def _get_stopwords():
    global _stop_words
    if _stop_words is None:
        try:
            _stop_words = set(stopwords.words("english"))
        except LookupError:
            download_nltk_resources()
            _stop_words = set(stopwords.words("english"))
    return _stop_words


def clean_text(text: str) -> str:
    """
    Remove URLs, mentions, hashtags symbol, emojis, punctuation, numbers,
    and extra whitespace. Lowercases the text.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)          # URLs
    text = re.sub(r"@\w+", " ", text)                        # mentions
    text = re.sub(r"#", " ", text)                            # hashtag symbol (keep word)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)                # emojis / non-ascii
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", " ", text)                          # numbers
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str):
    try:
        return word_tokenize(text)
    except LookupError:
        download_nltk_resources()
        return word_tokenize(text)


def remove_stopwords(tokens):
    sw = _get_stopwords()
    return [t for t in tokens if t not in sw]


def lemmatize_tokens(tokens):
    return [_lemmatizer.lemmatize(t) for t in tokens]


def stem_tokens(tokens):
    return [_stemmer.stem(t) for t in tokens]


def preprocess_for_bow_lstm(text: str) -> str:
    """Member 1 (Himas) pipeline: clean -> tokenize -> stopword removal -> lemmatize."""
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize_tokens(tokens)
    return " ".join(tokens)


def preprocess_for_tfidf_cnn(text: str) -> str:
    """Member 2 (Afrith) pipeline: clean -> lowercase -> tokenize -> stopword removal -> stem."""
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    tokens = remove_stopwords(tokens)
    tokens = stem_tokens(tokens)
    return " ".join(tokens)


def preprocess_for_bert(text: str) -> str:
    """
    Member 3 (Ilmaan) pipeline: clean -> lemmatize.
    NOTE: BERT's own WordPiece tokenizer handles subword tokenization, so we
    do NOT remove stopwords or apply aggressive stemming here -- that would
    destroy context BERT relies on.
    """
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    tokens = lemmatize_tokens(tokens)
    return " ".join(tokens)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true", help="Download required NLTK data")
    args = parser.parse_args()
    if args.download:
        download_nltk_resources()
        print("NLTK resources downloaded.")
    else:
        sample = "Official page!! DM for deals 🔥 https://t.co/xyz #promo @user123"
        print("Original:", sample)
        print("BoW/LSTM:", preprocess_for_bow_lstm(sample))
        print("TF-IDF/CNN:", preprocess_for_tfidf_cnn(sample))
        print("BERT:", preprocess_for_bert(sample))
