"""
preprocess.py
=============
Step 2 of the IR pipeline: turn a raw string into a clean list of terms.

This is where NLTK finally appears. The whole module is really just one
function, `preprocess()`, which runs five stages in order:

    1. lowercase        "Oil" and "oil" must become the same term
    2. tokenise         split the string into words
    3. filter           throw away punctuation and numbers
    4. filter           throw away stopwords ("the", "of", "is", ...)
    5. normalise        stem, so "exports"/"exporting" collapse to "export"

The single most important idea in this file: documents and queries BOTH go
through this same function. If a document becomes "export" but a query stays
"exporting", they will never match. Same pipeline on both sides, always.

Fill in the TODO blanks, then check your work with:

    python preprocess.py
"""

from typing import List

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# --------------------------------------------------------------------------
# One-time setup (given)
# --------------------------------------------------------------------------

def ensure_nltk_data() -> None:
    """Download the NLTK data files we need, if they aren't already present.

    Handy for teammates: they can just run the code instead of being told to
    run nltk.download() by hand first.
    """
    for pkg, path in [("punkt", "tokenizers/punkt"),
                      ("punkt_tab", "tokenizers/punkt_tab"),
                      ("stopwords", "corpora/stopwords"),
                      ("wordnet", "corpora/wordnet")]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)


ensure_nltk_data()

# Built once at import time, not inside preprocess(). Rebuilding these for
# every document would make indexing thousands of times slower.
STOPWORDS = set(stopwords.words("english"))   # 198 common English words
STEMMER = PorterStemmer()

# Tokens shorter than this are dropped. Single letters ("a", "b") carry
# almost no retrieval value but bloat the index.
MIN_TOKEN_LENGTH = 2


# --------------------------------------------------------------------------
# The pipeline
# --------------------------------------------------------------------------

def preprocess(text: str) -> List[str]:
    """Turn raw text into a list of index terms."""

    # TODO(1): Lowercase the text.
    #          Why first? So that "Oil", "OIL" and "oil" all end up as the
    #          same term, and so the stopword check below actually matches
    #          (the stopword list is stored in lowercase).
    text = text.lower()  # <-- your code here

    # TODO(2): Split the text into tokens using NLTK's word_tokenize().
    #          Note it is smarter than text.split(): it separates "yen," into
    #          "yen" and ",", which is exactly what we want before filtering.
    tokens = word_tokenize(text)  # <-- your code here

    # TODO(3): Keep only tokens that are purely alphabetic.
    #          Python strings have an .isalpha() method that returns True
    #          only if every character is a letter. This one test removes
    #          punctuation (",", ".") AND numbers ("5.2", "200,000") at once.
    #
    #          Design trade-off worth noting in your report: this also throws
    #          away "u.s." because of the dots. We accept that, since for
    #          topic-style retrieval the numbers and abbreviations are mostly
    #          noise. Mention the trade-off rather than pretending it is free.
    #          Hint: a list comprehension, [t for t in tokens if ...]
    tokens = [t for t in tokens if t.isalpha()]  # <-- your code here

    # TODO(4): Remove stopwords, and remove tokens shorter than
    #          MIN_TOKEN_LENGTH. Both conditions go in the same comprehension.
    #          Stopwords appear in nearly every document, so they cost a lot
    #          of index space while doing almost nothing to distinguish one
    #          document from another.
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) >= MIN_TOKEN_LENGTH]  # <-- your code here

    # TODO(5): Stem every remaining token with STEMMER.stem().
    #          Stemming chops words to a common root so that "export",
    #          "exports" and "exporting" all become one term.
    #          Do not be alarmed that stems are often not real words:
    #          "japanese" becomes "japanes". That is fine, because queries
    #          are stemmed by this same function and will match.
    tokens = [STEMMER.stem(t) for t in tokens]  # <-- your code here

    return tokens


def preprocess_query(query: str) -> List[str]:
    """Preprocess a user's search query.

    Deliberately just a call to preprocess(). It exists as a separate name so
    the query path is visible in the architecture diagram in your report, but
    it MUST stay identical in behaviour, for the reason in the module
    docstring above. Resist any temptation to treat queries differently.
    """
    return preprocess(query)


# --------------------------------------------------------------------------
# Self-check: run `python preprocess.py`
# --------------------------------------------------------------------------

if __name__ == "__main__":
    SENTENCE = "The U.S. Dollar rose 5.2 pct against the Japanese Yen, traders said."
    out = preprocess(SENTENCE)

    checks = [
        # (name, condition)
        ("returns a non-empty list",
         isinstance(out, list) and len(out) > 0),

        ("everything is lowercase",
         len(out) > 0 and all(t == t.lower() for t in out)),

        ("punctuation and numbers removed",
         len(out) > 0 and all(t.isalpha() for t in out)),

        ("stopwords removed ('the', 'against' gone)",
         len(out) > 0 and "the" not in out and "against" not in out),

        ("stemming applied ('running' and 'runs' agree)",
         len(preprocess("running")) > 0
         and preprocess("running") == preprocess("runs")),

        ("full output matches expected",
         out == ["dollar", "rose", "pct", "japanes", "yen", "trader", "said"]),
    ]

    print()
    for name, passed in checks:
        print("[%s] %s" % ("OK  " if passed else "FAIL", name))

    print("\ninput : ", SENTENCE)
    print("output: ", out)

    # Now run it on a real article, to see the two modules working together.
    try:
        from corpus import load_documents
        doc = load_documents("train")[0]
        print("\n--- real document", doc.doc_id, "---")
        print("raw   : ", doc.text[:120], "...")
        print("terms : ", preprocess(doc.text)[:20], "...")
    except Exception as exc:
        print("\n(skipped the real-document demo:", exc, ")")
