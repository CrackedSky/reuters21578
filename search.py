"""
search.py
=========
Step 4 of the IR pipeline: score documents against a query and rank them.

We use TF-IDF with cosine similarity. The idea in one paragraph:

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
    #          Use preprocess() from preprocess.py -- the SAME function the
    #          documents went through. If you use anything else here, the
    #          query terms will not match the indexed terms and every search
    #          returns nothing.
    query_terms = []  # <-- your code here

    # TODO(2): Build the query's term weights: {term: weight}.
    #          Same weighting as documents got in index.py:
    #              weight = tf_weight(count in query) * index.idf(term)
    #          Counter(query_terms) gives you the counts.
    #
    #          Then DROP any term whose weight is 0. A weight of 0 means
    #          idf is 0, i.e. the term is not in the collection at all
    #          (someone searched for "banana"). Keeping it would contribute
    #          nothing but would distort the query norm below.
    query_weights = {}  # <-- your code here

    # Nothing usable in the query -> no results. (Given.)
    if not query_weights:
        return []

    # TODO(3): Accumulate a raw score for every candidate document.
    #
    #          This is the heart of the whole system, and it is where the
    #          inverted index pays off: we only ever touch documents that
    #          actually contain a query term.
    #
    #            for each term and its query weight in query_weights:
    #                look up index.postings.get(term, {})
    #                for each doc_id and tf in that postings dict:
    #                    doc_weight = tf_weight(tf) * index.idf(term)
    #                    add (query_weight * doc_weight) onto scores[doc_id]
    #
    #          Use scores.get(doc_id, 0.0) to start a document at zero the
    #          first time you see it. Documents matching several query terms
    #          accumulate several contributions, which is exactly what we
    #          want -- matching two query words beats matching one.
    scores = {}  # <-- your code here

    # TODO(4): Normalise, turning raw overlap into a cosine similarity.
    #
    #          First compute the query norm, once:
    #              query_norm = sqrt(sum of w*w for w in query_weights.values())
    #
    #          Then divide every document's raw score by
    #              index.doc_norms[doc_id] * query_norm
    #
    #          Guard against a zero document norm (the 54 empty articles)
    #          by scoring those 0.0 instead of dividing by zero. In practice
    #          they never appear here, since an empty document is in nobody's
    #          postings list, but defensive code beats a crash in a demo.
    #
    #          Note the query norm is the same for every document, so it does
    #          not change the ORDER. We divide by it anyway so scores are on
    #          a 0-1 scale, which is much nicer to display.
    pass  # <-- your code here

    # TODO(5): Sort and return the best top_k.
    #
    #          Sort by score DESCENDING. Break ties by doc_id ASCENDING, so
    #          that runs are reproducible -- without a tie-break, equally
    #          scored documents come back in arbitrary dict order and your
    #          experiments will not reproduce.
    #
    #          Hint: sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    #          then slice [:top_k].
    return []  # <-- your code here


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

        ("toy: every score lies between 0 and 1",
         all(0.0 <= s <= 1.0 + 1e-9 for _, s in r_cat + r_sat + r_both)),
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
