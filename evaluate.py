"""
evaluate.py
===========
Step 5 of the IR pipeline: measure how good the retrieval actually is.

The problem: Reuters-21578 ships no relevance judgments, so there is no
official list of "documents that answer query X". What it does have is the
TOPICS category labels. So we borrow them:

    each category becomes a query
    the documents carrying that label are the relevant set

These are called *pseudo-qrels* ("qrels" = query relevance judgments). It is
a standard, defensible way to evaluate on a classification collection, and it
is the reason evaluate.py is allowed to look at doc.categories while
index.py is forbidden from doing so. Keeping that line clean is the whole
point: the index must never see the answers.

Honest caveat for your report: the README warns that many articles which
SHOULD carry a topic were never labelled. So the relevant sets have false
negatives, and a genuinely good hit can be scored as a miss. Your measured
precision is therefore a LOWER BOUND on true precision, not an exact value.

Fill in the TODO blanks, then check your work with:

    python evaluate.py
"""

from typing import Dict, List, Sequence, Set, Tuple

from corpus import Document, load_documents
from search import search, get_index

Result = Tuple[int, float]
Qrels = Dict[str, Set[int]]


# --------------------------------------------------------------------------
# Building the answer key
# --------------------------------------------------------------------------

def build_qrels(documents: Sequence[Document]) -> Qrels:
    """Map each category to the set of document ids carrying it.

    Result looks like: {"cocoa": {1, 57, 203, ...}, "wheat": {12, ...}, ...}

    TODO(1): Loop over the documents. For each category in doc.categories,
             add doc.doc_id to that category's set.

             A set, not a list: it makes the "is this document relevant?"
             test below O(1) instead of a scan, and it silently ignores any
             duplicate labelling.

             Hint: qrels.setdefault(category, set()).add(doc.doc_id)
             Documents with no categories simply contribute nothing.
    """
    qrels: Qrels = {}
    # <-- your code here
    return qrels


def category_to_query(category: str) -> str:
    """Turn a category code into a query string.

    Category codes are hyphenated, e.g. "money-fx", "money-supply". Splitting
    on the hyphen gives the tokeniser something it can work with.

    Possible improvement to write up later: cat-descriptions_120396.txt maps
    codes to real English ("Money/Foreign Exchange (MONEY-FX)"), which would
    give far more realistic multi-word queries than the bare code.
    """
    return category.replace("-", " ")


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def precision_at_k(results: List[Result], relevant: Set[int], k: int) -> float:
    """Of the top k documents returned, what fraction were relevant?

    "I looked at the first 10 hits; how many were any good?" This is the
    metric that matches what a user actually experiences.

    TODO(2): Take the first k results, count how many of their doc_ids are
             in `relevant`, and divide by k.

             Divide by k, NOT by len(results). If a query returns only 3
             documents and you asked for P@10, the other 7 slots were still
             shown to the user as empty, so they count against you.
             Return 0.0 if k is 0.
    """
    return 0.0  # <-- your code here


def recall_at_k(results: List[Result], relevant: Set[int], k: int) -> float:
    """Of all the relevant documents that exist, what fraction did we find?

    Precision asks "was what I got any good?"; recall asks "did I miss
    anything?". They pull against each other, which is why both get reported.

    TODO(3): Count relevant documents in the top k, then divide by the TOTAL
             number of relevant documents, len(relevant).
             Return 0.0 if `relevant` is empty, to avoid dividing by zero.
    """
    return 0.0  # <-- your code here


def average_precision(results: List[Result], relevant: Set[int]) -> float:
    """Average of the precision values at each relevant document found.

    This rewards putting relevant documents EARLY, which precision@k alone
    does not. Worked example, with 3 relevant documents in the collection
    and relevant hits landing at ranks 1, 3 and 5:

        rank 1  relevant   -> precision so far = 1/1 = 1.000
        rank 2  not
        rank 3  relevant   -> precision so far = 2/3 = 0.667
        rank 4  not
        rank 5  relevant   -> precision so far = 3/5 = 0.600

        AP = (1.000 + 0.667 + 0.600) / 3 = 0.756

    TODO(4): Walk the results in rank order, keeping a running count of hits
             so far. Every time you hit a relevant document, add
             (hits so far / current rank) to a running total. At the end
             divide that total by len(relevant).

             Rank is 1-based, so enumerate(results, start=1).
             Return 0.0 if `relevant` is empty.

             Note we divide by the total relevant in the COLLECTION, not by
             how many we retrieved. So if 55 documents are relevant and you
             only retrieve the top 100, any relevant document past rank 100
             counts as a zero. That is intentional -- missing documents
             should hurt your score.
    """
    return 0.0  # <-- your code here


# --------------------------------------------------------------------------
# Running the whole experiment
# --------------------------------------------------------------------------

def evaluate_all(index, qrels: Qrels, top_k: int = 100,
                 verbose: bool = False) -> Dict[str, float]:
    """Run every category as a query and average the results.

    Returns a dict with "map", "p@10" and "r@10", each averaged over queries.
    MAP (Mean Average Precision) is just the mean of average_precision over
    all queries -- the single number most IR papers lead with.

    TODO(5): For each category in sorted(qrels):
               - turn it into a query with category_to_query()
               - run search(index, query, top_k)
               - append average_precision(...) to `aps`
               - append precision_at_k(..., 10) to `p10s`
               - append recall_at_k(..., 10) to `r10s`
               - if verbose, print a line per category so you can see which
                 topics do well and which do badly

             Sorted, so runs are reproducible and comparable.
             Then return the mean of each list.
    """
    aps: List[float] = []
    p10s: List[float] = []
    r10s: List[float] = []

    # <-- your code here

    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
    return {"map": mean(aps), "p@10": mean(p10s), "r@10": mean(r10s)}


# --------------------------------------------------------------------------
# Self-check: run `python evaluate.py`
# --------------------------------------------------------------------------

if __name__ == "__main__":
    # --- Part 1: metrics on a hand-made ranking --------------------------
    # 5 results; documents 1, 3 and 5 are the relevant ones.
    fake = [(1, 0.9), (2, 0.8), (3, 0.7), (4, 0.6), (5, 0.5)]
    rel = {1, 3, 5}

    metric_checks = [
        ("P@1 == 1.0", abs(precision_at_k(fake, rel, 1) - 1.0) < 1e-9),
        ("P@3 == 2/3", abs(precision_at_k(fake, rel, 3) - 2 / 3) < 1e-9),
        ("P@5 == 0.6", abs(precision_at_k(fake, rel, 5) - 0.6) < 1e-9),
        ("R@5 == 1.0", abs(recall_at_k(fake, rel, 5) - 1.0) < 1e-9),
        ("R@1 == 1/3", abs(recall_at_k(fake, rel, 1) - 1 / 3) < 1e-9),
        ("AP == 0.7556", abs(average_precision(fake, rel) - 0.75556) < 1e-4),
        ("empty relevant set gives 0.0, not a crash",
         average_precision(fake, set()) == 0.0),
    ]

    print()
    for name, passed in metric_checks:
        print("[%s] %s" % ("OK  " if passed else "FAIL", name))

    if not all(p for _, p in metric_checks):
        print("\nMetrics failing -- fix those before running the real thing.")
        raise SystemExit(1)

    # --- Part 2: the real experiment -------------------------------------
    documents = load_documents("train")
    qrels = build_qrels(documents)
    index = get_index("train")

    qrels_checks = [
        ("115 categories appear in the training set", len(qrels) == 115),
        ("55 documents are labelled 'cocoa'", len(qrels.get("cocoa", set())) == 55),
        ("197 documents are labelled 'ship'", len(qrels.get("ship", set())) == 197),
    ]

    print()
    for name, passed in qrels_checks:
        print("[%s] %s" % ("OK  " if passed else "FAIL", name))

    if not all(p for _, p in qrels_checks):
        raise SystemExit(1)

    # A few individual categories, so you can see the spread.
    print("\nPer-category results:")
    print("  %-10s %7s %7s %7s %7s" % ("category", "#rel", "P@10", "R@10", "AP"))
    for cat in ["cocoa", "gold", "coffee", "sugar", "ship"]:
        res = search(index, category_to_query(cat), 100)
        r = qrels[cat]
        print("  %-10s %7d %7.3f %7.3f %7.4f"
              % (cat, len(r), precision_at_k(res, r, 10),
                 recall_at_k(res, r, 10), average_precision(res, r)))

    overall = evaluate_all(index, qrels, top_k=100)
    print("\nOverall over all %d categories:" % len(qrels))
    print("  MAP  = %.4f" % overall["map"])
    print("  P@10 = %.4f" % overall["p@10"])
    print("  R@10 = %.4f" % overall["r@10"])

    print()
    print("[%s] MAP is about 0.3487" % (
        "OK  " if abs(overall["map"] - 0.3487) < 0.001 else "FAIL"))

    print("\nThis is your baseline. Every improvement you try later gets")
    print("compared against these numbers -- that comparison IS the")
    print("experimental evaluation section of the report.")
