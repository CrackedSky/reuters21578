"""
index.py
========
Step 3 of the IR pipeline: build the inverted index.

An *inverted* index maps each term to the documents containing it:

    "cocoa" -> {doc 1: 4 times, doc 57: 2 times, ...}
    "wheat" -> {doc 12: 1 time, ...}

It is called "inverted" because the obvious way round would be
document -> words. Going term -> documents is what makes search fast: to
answer a query we only look at documents that actually contain a query
term, instead of scanning all 9,603 articles.

We store five things:

  postings        term -> {doc_id: how many times the term occurs in that doc}
  doc_norms       doc_id -> TF-IDF vector norm (for cosine similarity)
  titles          doc_id -> title, purely so results are readable on screen
  doc_lengths     doc_id -> number of preprocessed terms (for BM25)
  avg_doc_length  average preprocessed document length (for BM25)

Fill in the TODO blanks, then check your work with:

    python index.py

Note the real index takes roughly 35 seconds to build, because stemming
10,000 articles is slow. That is exactly why save()/load() exist.
"""

import os
import math
import pickle
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List

from corpus import Document, load_documents
from preprocess import preprocess

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------
# Weighting helper (given)
# --------------------------------------------------------------------------

def tf_weight(tf: int) -> float: # tf: term frequency
    """Sublinear term frequency weight: 1 + log10(tf).

    Why not just use the raw count? Because a document mentioning "oil" 20
    times is not 20 times more about oil than one mentioning it once. The
    logarithm damps that down: tf=1 -> 1.0, tf=10 -> 2.0, tf=100 -> 3.0.
    """
    return 1.0 + math.log10(tf) if tf > 0 else 0.0


# --------------------------------------------------------------------------
# The index
# --------------------------------------------------------------------------

@dataclass
class InvertedIndex:
    # df: document frequency
    # idf: inverse document frequency (log10(N / df), where N = 9603)
    postings: Dict[str, Dict[int, int]]
    doc_norms: Dict[int, float]
    titles: Dict[int, str]
    doc_lengths: Dict[int, int] = field(default_factory=dict)
    avg_doc_length: float = 0.0

    @property
    def num_docs(self) -> int:
        """Total documents in the collection (called N in the IDF formula)."""
        return len(self.doc_norms)

    def document_frequency(self, term: str) -> int:
        """How many documents contain `term`? (df)

        TODO(3): The postings entry for a term is a dict {doc_id: tf}, so the
                 number of documents containing it is just how many keys that
                 dict has. Return 0 if the term is not in the index at all.
                 Hint: self.postings.get(term, {}) avoids a KeyError.
        """
        return len(self.postings.get(term, {}))

    def idf(self, term: str) -> float:
        """Inverse document frequency: log10(N / df).

        This is the "rare words matter more" part of TF-IDF. A term in only
        59 of 9,603 documents ("cocoa") is highly informative. A term in
        1,859 of them ("bank") barely narrows anything down, so it should
        score lower.

        TODO(4): Get df via self.document_frequency(term). If df is 0 the
                 term is not in the collection, so return 0.0 - otherwise you
                 would divide by zero. Otherwise return log10(N / df), using
                 self.num_docs for N and math.log10().
        """
        return 0.0 if self.document_frequency(term) == 0 else math.log10(self.num_docs / self.document_frequency(term))


# --------------------------------------------------------------------------
# Building
# --------------------------------------------------------------------------

def build_index(documents: List[Document]) -> InvertedIndex:
    """Turn a list of Documents into an InvertedIndex."""

    postings: Dict[str, Dict[int, int]] = {}
    titles = {d.doc_id: d.title for d in documents}

    # Every document gets a norm entry up front, including ones that end up
    # with no terms at all. 54 articles in the training set are empty after
    # preprocessing (they are stubs or pure numbers). They can never be
    # retrieved, but they must still count towards N or every IDF is wrong.
    doc_norms = {d.doc_id: 0.0 for d in documents}
    # BM25 needs each document length after the SAME preprocessing pipeline.
    doc_lengths = {d.doc_id: 0 for d in documents}

    for doc in documents:
        # Only doc.text -- title + body. Never doc.categories, which is the
        # answer key we grade against later.
        terms = preprocess(doc.text)
        doc_lengths[doc.doc_id] = len(terms)

        # TODO(1): Count how many times each term appears in THIS document.
        #          collections.Counter is imported for you and does exactly
        #          this: Counter(["a", "b", "a"]) gives {"a": 2, "b": 1}.
        counts = Counter(terms)  # <-- your code here

        # TODO(2): Merge those counts into `postings`.
        #          For each term and its count, record it under
        #          postings[term][doc.doc_id] = count
        #          The catch: postings[term] may not exist yet. Use
        #          postings.setdefault(term, {}) which returns the existing
        #          dict or inserts a fresh empty one.
        #          Hint: for term, count in counts.items():
        for term, count in counts.items():
            if term not in postings:
                postings[term] = {}
            postings[term][doc.doc_id] = count

    avg_doc_length = (sum(doc_lengths.values()) / len(documents)
                      if documents else 0.0)
    index = InvertedIndex(postings=postings, doc_norms=doc_norms, titles=titles,
                          doc_lengths=doc_lengths,
                          avg_doc_length=avg_doc_length)

    # TODO(5): Fill in the document norms.
    #
    #   Why: without this, long documents win every search simply by being
    #   long and containing more words. Dividing each document's score by its
    #   norm (done later in search.py) cancels that out. This is the
    #   "cosine normalisation" step.
    #
    #   The norm of a document is the square root of the sum, over every term
    #   in that document, of (weight of the term in that document) squared,
    #   where weight = tf_weight(tf) * index.idf(term).
    #
    #   Work term by term, because idf() only needs computing once per term:
    #
    #     for term, plist in index.postings.items():
    #         weight_idf = index.idf(term)
    #         for doc_id, tf in plist.items():
    #             accumulate (tf_weight(tf) * weight_idf) ** 2 into
    #             index.doc_norms[doc_id]
    #
    #   Then afterwards, replace every accumulated total with its square root
    #   (math.sqrt).
    #
    #   This must happen AFTER the loop above, because idf() needs the
    #   complete postings for the whole collection before it means anything.
    for term, plist in index.postings.items():
        weight_idf = index.idf(term)
        for doc_id, tf in plist.items():
            index.doc_norms[doc_id] += (tf_weight(tf) * weight_idf) ** 2

    # Take the square root of each accumulated total
    for doc_id in index.doc_norms:
        index.doc_norms[doc_id] = math.sqrt(index.doc_norms[doc_id])

    return index


# --------------------------------------------------------------------------
# BM25 length-stat compatibility helper
# --------------------------------------------------------------------------

def ensure_length_stats(index: InvertedIndex) -> InvertedIndex:
    """Ensure an index has the document-length statistics BM25 needs.

    Older cached index_*.pkl files were created before `doc_lengths` and
    `avg_doc_length` existed. Their postings still contain every raw term
    frequency, so we can reconstruct the lengths without reparsing Reuters.
    """
    if (getattr(index, "doc_lengths", None)
            and getattr(index, "avg_doc_length", 0.0) > 0.0):
        return index

    doc_lengths = {doc_id: 0 for doc_id in index.doc_norms}
    for plist in index.postings.values():
        for doc_id, tf in plist.items():
            doc_lengths[doc_id] += tf

    index.doc_lengths = doc_lengths
    index.avg_doc_length = (sum(doc_lengths.values()) / index.num_docs
                            if index.num_docs else 0.0)
    return index


# --------------------------------------------------------------------------
# Saving and loading (given -- boilerplate, but it saves you 35s every run)
# --------------------------------------------------------------------------

def save_index(index: InvertedIndex, filename: str) -> None:
    with open(os.path.join(DATA_DIR, filename), "wb") as f:
        pickle.dump(index, f)


def load_index(filename: str) -> InvertedIndex:
    with open(os.path.join(DATA_DIR, filename), "rb") as f:
        index = pickle.load(f)
    return ensure_length_stats(index)


# --------------------------------------------------------------------------
# Self-check: run `python index.py`
# --------------------------------------------------------------------------

if __name__ == "__main__":
    # --- Part 1: a toy corpus small enough to verify by hand -------------
    # After preprocessing these become:
    #   doc 0: ['cat', 'sat', 'mat']     ("the" and "on" are stopwords)
    #   doc 1: ['dog', 'sat', 'log']
    #   doc 2: ['cat', 'dog']            ("cats"/"dogs" stem to cat/dog)
    toy_docs = [
        Document(0, "", "the cat sat on the mat", [], "TRAIN", "YES"),
        Document(1, "", "the dog sat on the log", [], "TRAIN", "YES"),
        Document(2, "", "cats and dogs", [], "TRAIN", "YES"),
    ]
    toy = build_index(toy_docs)

    toy_checks = [
        ("toy: 3 documents", toy.num_docs == 3),
        ("toy: 5 distinct terms", len(toy.postings) == 5),
        ("toy: 'cat' appears in docs 0 and 2", toy.postings.get("cat") == {0: 1, 2: 1}),
        ("toy: df('sat') == 2", toy.document_frequency("sat") == 2),
        ("toy: df('mat') == 1", toy.document_frequency("mat") == 1),
        ("toy: df of unknown term == 0", toy.document_frequency("banana") == 0),
        ("toy: idf of unknown term == 0.0", toy.idf("banana") == 0.0),
        ("toy: rarer term has higher idf", toy.idf("mat") > toy.idf("cat")),
        ("toy: all norms > 0", all(v > 0 for v in toy.doc_norms.values())),
        ("toy: document lengths are 3, 3, 2",
         toy.doc_lengths == {0: 3, 1: 3, 2: 2}),
        ("toy: average document length is 8/3",
         abs(toy.avg_doc_length - 8 / 3) < 1e-9),
    ]

    print()
    for name, passed in toy_checks:
        print("[%s] %s" % ("OK  " if passed else "FAIL", name))

    if not all(p for _, p in toy_checks):
        print("\nToy corpus failing -- fix that before building the real index.")
        raise SystemExit(1)

    # --- Part 2: the real collection ------------------------------------
    print("\nBuilding the real index (about 35 seconds)...")
    real = build_index(load_documents("train"))

    real_checks = [
        ("real: 9,603 documents indexed", real.num_docs == 9603),
        ("real: vocabulary of 19,229 terms", len(real.postings) == 19229),
        ("real: df('cocoa') == 59", real.document_frequency("cocoa") == 59),
        ("real: df('bank') == 1859", real.document_frequency("bank") == 1859),
        ("real: idf('cocoa') is about 2.212", abs(real.idf("cocoa") - 2.212) < 0.01),
        ("real: rare 'cocoa' outranks common 'bank'", real.idf("cocoa") > real.idf("bank")),
        ("real: doc 1 norm is about 27.395", abs(real.doc_norms[1] - 27.395) < 0.01),
    ]

    print()
    for name, passed in real_checks:
        print("[%s] %s" % ("OK  " if passed else "FAIL", name))

    if all(p for _, p in real_checks):
        save_index(real, "index_train.pkl")
        print("\nSaved to index_train.pkl -- search.py will load this instead")
        print("of rebuilding, so you only pay the 35 seconds once.")