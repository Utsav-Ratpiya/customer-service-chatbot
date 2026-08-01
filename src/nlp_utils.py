"""
nlp_utils.py
------------
Lightweight text preprocessing utilities for the customer service chatbot.

Deliberately dependency-light (no NLTK/spaCy downloads required) so the
project runs anywhere with just scikit-learn installed. Implements:
    - text cleaning / normalization
    - tokenization
    - stopword filtering (negations preserved on purpose)
    - a small rule-based suffix stripper (light stemming)
"""

import json
import os
import re
import string

# A compact custom stopword list. Deliberately EXCLUDES negation words
# ("no", "not", "n't", "never", "without") because dropping them can flip
# the meaning of a customer query (e.g. "my order has NOT arrived").
STOPWORDS = {
    "a", "an", "the", "is", "am", "are", "was", "were", "be", "been", "being",
    "i", "me", "my", "myself", "we", "our", "ours", "you", "your", "yours",
    "he", "him", "his", "she", "her", "it", "its", "they", "them", "their",
    "this", "that", "these", "those", "of", "at", "by", "for", "with", "about",
    "to", "from", "in", "on", "up", "down", "again", "further", "then", "once",
    "here", "there", "all", "any", "both", "each", "few", "more", "most",
    "other", "some", "such", "than", "too", "very", "s", "t", "can", "will",
    "just", "should", "now", "do", "does", "did", "doing", "and", "or", "if",
    "please", "kindly",
}

# A handful of common English suffixes, longest first, for a very small
# rule-based stemmer. This is not a full Porter stemmer, but it merges
# common variants like "shipping/shipped/ships" -> "ship" reasonably well.
_SUFFIXES = ["ing", "edly", "ed", "ly", "ies", "ied", "es", "s"]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INTENTS_PATH = os.path.join(BASE_DIR, "data", "intents.json")

# A set of known words extracted from the training data patterns
VOCABULARY = set()

def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
        
    return previous_row[-1]

def load_vocabulary():
    global VOCABULARY
    VOCABULARY.clear()
    if not os.path.exists(INTENTS_PATH):
        return
    try:
        with open(INTENTS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            for intent in data.get("intents", []):
                for pattern in intent.get("patterns", []):
                    pat_clean = pattern.lower().strip()
                    pat_clean = re.sub(r"[^a-z0-9\s]", " ", pat_clean)
                    for word in pat_clean.split():
                        if len(word) > 2 and word not in STOPWORDS:
                            VOCABULARY.add(word)
    except Exception:
        pass

# Initialize vocabulary
load_vocabulary()

def spell_correct_word(word: str) -> str:
    """Correct simple typos in a word by matching it to the closest word in VOCABULARY."""
    if not VOCABULARY or word in VOCABULARY or len(word) <= 3 or word.isdigit():
        return word
        
    # Check for words with edit distance 1
    for vocab_word in VOCABULARY:
        if abs(len(vocab_word) - len(word)) > 1:
            continue
        if levenshtein_distance(word, vocab_word) == 1:
            return vocab_word
            
    # Check for words with edit distance 2 for longer words
    if len(word) > 5:
        best_word = None
        best_dist = 99
        for vocab_word in VOCABULARY:
            if abs(len(vocab_word) - len(word)) > 2:
                continue
            dist = levenshtein_distance(word, vocab_word)
            if dist == 2 and dist < best_dist:
                best_word = vocab_word
                best_dist = dist
        if best_word:
            return best_word
            
    return word



def clean_text(text: str) -> str:
    """Lowercase, strip punctuation/extra whitespace, keep apostrophes for
    contractions (don't, can't, won't) before they get expanded."""
    text = text.lower().strip()
    text = _expand_contractions(text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


_CONTRACTIONS = {
    "won't": "will not", "can't": "can not", "n't": " not",
    "'re": " are", "'s": " is", "'d": " would", "'ll": " will",
    "'t": " not", "'ve": " have", "'m": " am",
}


def _expand_contractions(text: str) -> str:
    for pattern, replacement in _CONTRACTIONS.items():
        text = text.replace(pattern, replacement)
    return text


def simple_stem(word: str) -> str:
    """Strip common suffixes off a word. Falls back to the original word
    if stripping would make it too short to be meaningful."""
    for suf in _SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            return word[: -len(suf)]
    return word


def tokenize(text: str, remove_stopwords: bool = True, stem: bool = True):
    """Turn raw text into a list of normalized tokens."""
    cleaned = clean_text(text)
    tokens = cleaned.split()
    # Correct spelling typos
    tokens = [spell_correct_word(t) for t in tokens]
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    if stem:
        tokens = [simple_stem(t) for t in tokens]
    return tokens


def preprocess_for_vectorizer(text: str) -> str:
    """Full pipeline that returns a single normalized string, suitable as
    input to a scikit-learn TfidfVectorizer."""
    tokens = tokenize(text, remove_stopwords=True, stem=True)
    return " ".join(tokens)


def extract_order_id(text: str):
    """Rule-based entity extraction: pull an order ID out of free text.
    Matches patterns like ORD12345, #12345, ORDER-98765, or a bare 5-10
    digit number."""
    patterns = [
        r"\b(ORD[- ]?\d{4,10})\b",
        r"\b(ORDER[- ]?\d{4,10})\b",
        r"#(\d{4,10})\b",
        r"\b(\d{5,10})\b",
    ]
    upper_text = text.upper()
    for pat in patterns:
        match = re.search(pat, upper_text)
        if match:
            return match.group(1).replace(" ", "").replace("--", "-")
    return None


def extract_email(text: str):
    """Rule-based entity extraction for an email address."""
    match = re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)
    return match.group(0) if match else None
