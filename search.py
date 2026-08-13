"""
search.py
=========
Step 4 of the IR pipeline: score documents against a query and rank them.

This module supports two ranking models:

  1. TF-IDF with cosine similarity (the original baseline)
  2. BM25 (an additional ranking method for comparison)

The TF-IDF idea in one paragraph:

  Think of each document as a point in space, with one axis per term. A
  document's position along the "cocoa" axis is how strongly it is about
  cocoa (that is tf_weight(tf) * idf). A query is a point in the same space.
  Documents close to the query are relevant. "Cosine similarity" measures
  the ANGLE between them rather than the distance, which is what makes a
  short article and a long article about the same topic score alike.

The score works out to:

                 sum over shared terms of (query weight * document weight)
  score(d, q) = -----------------------------------------------------------
                          document norm  *  query norm

The numerator rewards overlap. The denominator is the normalisation that
stops long documents winning automatically. Scores land between 0 and 1,
where 1.0 means the document contains exactly the query terms in exactly
the same proportions.

Fill in the TODO blanks, then check your work with:

    python search.py
"""

import math
from collections import Counter
from typing import List, Tuple

from corpus import Document, load_documents
from preprocess import preprocess
from index import (InvertedIndex, tf_weight, build_index,
                   load_index, save_index)

# A search result is a (doc_id, score) pair.
Result = Tuple[int, float]


# --------------------------------------------------------------------------
# The ranking function
# --------------------------------------------------------------------------

def search(index: InvertedIndex, query: str, top_k: int = 10) -> List[Result]:
    """Return the top_k (doc_id, score) pairs, best first."""

    # TODO(1): Preprocess the query into a list of terms.
    query_terms = preprocess(query)

    # TODO(2): Build the query's term weights.
    query_counts = Counter(query_terms)
    query_weights = {}

    for term, count in query_counts.items():
        weight = tf_weight(count) * index.idf(term)

        if weight != 0:
            query_weights[term] = weight

    # Nothing usable in the query -> no results.
    if not query_weights:
        return []

    # TODO(3): Accumulate raw scores for candidate documents.
    scores = {}

    for term, query_weight in query_weights.items():
        postings = index.postings.get(term, {})

        for doc_id, tf in postings.items():
            doc_weight = tf_weight(tf) * index.idf(term)

            scores[doc_id] = (
                scores.get(doc_id, 0.0)
                + query_weight * doc_weight
            )

    # TODO(4): Cosine normalisation.
    query_norm = math.sqrt(
        sum(weight * weight for weight in query_weights.values())
    )

    for doc_id in scores:
        doc_norm = index.doc_norms[doc_id]

        if doc_norm == 0.0:
            scores[doc_id] = 0.0
        else:
            scores[doc_id] = scores[doc_id] / (doc_norm * query_norm)

    # TODO(5): Sort best score first.
    return sorted(
        scores.items(),
        key=lambda kv: (-kv[1], kv[0])
    )[:top_k]


# --------------------------------------------------------------------------
# BM25 ranking
# --------------------------------------------------------------------------

def bm25_idf(index: InvertedIndex, term: str) -> float:
    """Robertson/Sparck Jones IDF variant used by BM25.

    The +1 keeps IDF non-negative even for very common terms:
        ln(1 + (N - df + 0.5) / (df + 0.5))
    """
    df = index.document_frequency(term)
    if df == 0:
        return 0.0
    return math.log(1.0 + (index.num_docs - df + 0.5) / (df + 0.5))


def search_bm25(index: InvertedIndex, query: str, top_k: int = 10,
                 k1: float = 1.5, b: float = 0.75) -> List[Result]:
    """Return the top_k documents ranked with BM25.

    k1 controls term-frequency saturation. b controls document-length
    normalisation (0 = none, 1 = full). Common defaults are around
    k1=1.2-2.0 and b=0.75.
    """
    query_terms = preprocess(query)
    if not query_terms or index.avg_doc_length <= 0.0:
        return []

    scores = {}

    # For short IR queries, standard BM25 usually treats each distinct query
    # term once. Repeating a word in the query therefore does not multiply it.
    for term in set(query_terms):
        idf = bm25_idf(index, term)
        if idf == 0.0:
            continue

        for doc_id, tf in index.postings.get(term, {}).items():
            doc_length = index.doc_lengths.get(doc_id, 0)
            length_norm = 1.0 - b + b * (doc_length / index.avg_doc_length)
            denominator = tf + k1 * length_norm
            term_score = idf * (tf * (k1 + 1.0)) / denominator
            scores[doc_id] = scores.get(doc_id, 0.0) + term_score

    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]


# --------------------------------------------------------------------------
# Display helper (given)
# --------------------------------------------------------------------------

def format_results(index: InvertedIndex, results: List[Result]) -> str:
    """Render results the way the assignment brief shows them."""
    if not results:
        return "  No matching documents."
    lines = []
    for rank, (doc_id, score) in enumerate(results, start=1):
        title = index.titles.get(doc_id, "") or "(no title)"
        lines.append("  %2d. %-58s Score: %.4f  [id %d]"
                     % (rank, title[:58], score, doc_id))
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Shared helper so every module reuses one cached index (given)
# --------------------------------------------------------------------------

def get_index(split: str = "train") -> InvertedIndex:
    """Load the saved index, building and saving it the first time."""
    filename = "index_%s.pkl" % split
    try:
        return load_index(filename)
    except (FileNotFoundError, EOFError):
        print("No cached index, building (about 35 seconds)...")
        index = build_index(load_documents(split))
        save_index(index, filename)
        return index


# --------------------------------------------------------------------------
# Self-check: run `python search.py`
# --------------------------------------------------------------------------

if __name__ == "__main__":
    # --- Part 1: the toy corpus, verifiable by hand ----------------------
    #   doc 0: ['cat', 'sat', 'mat']
    #   doc 1: ['dog', 'sat', 'log']
    #   doc 2: ['cat', 'dog']
    toy_docs = [
        Document(0, "", "the cat sat on the mat", [], "TRAIN", "YES"),
        Document(1, "", "the dog sat on the log", [], "TRAIN", "YES"),
        Document(2, "", "cats and dogs", [], "TRAIN", "YES"),
    ]
    toy = build_index(toy_docs)

    r_cat = search(toy, "cat")
    r_sat = search(toy, "sat")
    r_both = search(toy, "cat dog")
    r_bm25 = search_bm25(toy, "cat dog")

    toy_checks = [
        ("toy: 'cat' finds docs 2 and 0, in that order",
         [d for d, _ in r_cat] == [2, 0]),

        # Doc 2 beats doc 0 even though both contain 'cat' exactly once,
        # because doc 2 is shorter -- 'cat' is a bigger share of it.
        ("toy: 'cat' scores doc 2 at about 0.707",
         len(r_cat) == 2 and abs(r_cat[0][1] - 0.7071) < 0.001),

        ("toy: 'sat' finds docs 0 and 1, tie broken by doc_id",
         [d for d, _ in r_sat] == [0, 1]),

        # Doc 2 IS exactly ['cat', 'dog'] -- same terms, same proportions as
        # the query -- so the angle between them is zero and cosine is 1.0.
        ("toy: 'cat dog' scores doc 2 at exactly 1.0",
         len(r_both) == 3 and abs(r_both[0][1] - 1.0) < 1e-9),

        ("toy: unknown word returns nothing",
         search(toy, "banana") == []),

        ("toy: every TF-IDF score lies between 0 and 1",
         all(0.0 <= s <= 1.0 + 1e-9 for _, s in r_cat + r_sat + r_both)),
        ("toy: BM25 'cat dog' ranks exact two-term doc 2 first",
         r_bm25 and r_bm25[0][0] == 2),
        ("toy: BM25 unknown word returns nothing",
         search_bm25(toy, "banana") == []),
    ]

    print()
    for name, passed in toy_checks:
        print("[%s] %s" % ("OK  " if passed else "FAIL", name))

    if not all(p for _, p in toy_checks):
        print("\nToy corpus failing -- fix that before touching the real index.")
        raise SystemExit(1)

    # --- Part 2: the real collection -------------------------------------
    index = get_index("train")
    real = search(index, "cocoa", 5)

    real_checks = [
        ("real: 'cocoa' returns 5 results", len(real) == 5),
        ("real: top hit is document 10471", real and real[0][0] == 10471),
        ("real: top score is about 0.4767",
         real and abs(real[0][1] - 0.4767) < 0.001),
        ("real: scores come back in descending order",
         all(real[i][1] >= real[i + 1][1] for i in range(len(real) - 1))),
    ]

    print()
    for name, passed in real_checks:
        print("[%s] %s" % ("OK  " if passed else "FAIL", name))

    # --- Part 3: eyeball some real searches ------------------------------
    for demo in ["cocoa", "coffee exports brazil", "interest rate"]:
        print("\nQuery: %r" % demo)
        print(format_results(index, search(index, demo, 5)))