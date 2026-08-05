# KIT719 Project 1 — Session Handover

Last updated: 2026-08-04

## What this project is

Build an Information Retrieval (IR) system over the Reuters-21578 corpus using
Python + NLTK, as a **command-line application**. Assignment brief lives at
`../KIT719 Project 1.docx`. Worth 25%, group project (2–3 people).

Marks breakdown: architecture/implementation 12, evaluation/analysis 8,
report writing 3, problem/data description 2.

**Admin not yet confirmed done:** the brief requires emailing
quan.bai@utas.edu.au before end of Week 3 with group member names, student IDs,
and the chosen dataset.

## Key design decisions already made

**Use the ModApte split.** Reuters-21578 has no relevance judgments, so the
predefined splits get reinterpreted for IR: each TOPICS category becomes a
query topic, and documents carrying that label are the relevant set
(pseudo-qrels). TRAIN is for development and tuning; TEST is held out for
final numbers. ModLewis was rejected because it includes `TOPICS="NO"` docs
that have no labels, which would sit unjudged in the rankings. ModHayes was
rejected because the README documents known defects (near-duplicates,
chronological burstiness).

**Index only `<TITLE>` and `<BODY>`.** Never index the `<TOPICS>` /
`<PLACES>` / `<ORGS>` fields — they are the answer key used by evaluate.py.
Leaking them into searchable text produces a near-perfect, meaningless score.
This is enforced by the `Document.text` property in corpus.py.

**Scope deliberately kept small.** BM25, RM3 query expansion, WordNet synonym
retrieval and spelling correction were discussed and then set aside. Plain
TF-IDF is a complete, legitimate system. Revisit the advanced techniques only
after the basic pipeline works end to end.

## Verified facts about the data

All confirmed by running against the actual `.sgm` files, and consistent with
README.txt section VIII.

| Quantity | Value |
|---|---|
| Total articles | 21,578 |
| ModApte TRAIN (`LEWISSPLIT="TRAIN"` + `TOPICS="YES"`) | 9,603 |
| ModApte TEST (`LEWISSPLIT="TEST"` + `TOPICS="YES"`) | 3,299 |
| TRAIN docs with ≥1 category | 7,775 |
| TEST docs with ≥1 category | 3,019 |
| Categories present in both TRAIN and TEST | 90 (the standard R(90) benchmark) |
| Mean doc length (title+body) | 124 words (median 84) |
| TEST docs with no `<BODY>` | 290 |

Category frequency is extremely skewed: `earn` has 1,087 relevant test docs
and `acq` has 719, but 29 categories have 3 or fewer. That head/tail contrast
is the most interesting evaluation result available in this dataset.

**Trap the README warns about:** the `TOPICS` attribute is *not* the same as
whether an article has category labels. A `TOPICS="YES"` article can have zero
labels. Split filtering must use the attribute, not the label list.

**Useful file:** `cat-descriptions_120396.txt` maps category codes to natural
language, e.g. `Money/Foreign Exchange (MONEY-FX)`. Use it later to build
realistic multi-word queries instead of raw codes like `money-fx`.

## Planned modules (6)

| # | File | Job | Status |
|---|---|---|---|
| 1 | `corpus.py` | Parse `.sgm` files into `Document` objects; filter by split | Done, one bug outstanding |
| 2 | `preprocess.py` | Raw text → clean list of terms (NLTK) | Skeleton written, blanks unfilled |
| 3 | `index.py` | Build inverted index (term → documents containing it) | Not started |
| 4 | `search.py` | Score and rank documents with TF-IDF | Not started |
| 5 | `evaluate.py` | Precision/recall against the category pseudo-qrels | Not started |
| 6 | `cli.py` | argparse front end tying it together | Not started |

Data flow: `corpus.py` loads documents → `preprocess.py` cleans them →
`index.py` builds the index. At search time a query goes through the *same*
`preprocess.py` → `search.py` consults the index → ranked results.
`evaluate.py` automates that path over many queries. Only `cli.py` imports
everything else.

## Working method (keep doing this)

Claude writes each module as a **skeleton with numbered TODO blanks** and
explanatory comments, plus a `if __name__ == "__main__":` self-check block
containing verified expected values. The user fills the blanks, runs the file,
and gets OK/FAIL per check. Claude then reviews the filled-in code.

Build modules in order 1, 2, 3, 4, 6 to get a working search you can type
into; add 5 afterwards.

## Outstanding issue — fix this first next session

`corpus.py` line 139 has an extra pair of square brackets:

```python
categories = [D_RE.findall(topics)]   # wrong — produces [['cocoa']]
categories = D_RE.findall(topics)     # correct
```

`findall()` already returns a list, so wrapping it creates a nested list. The
practical damage: an article with no topics becomes `[[]]`, which is truthy
because it is a one-element list, so every one of the 9,603 articles counts as
having categories. The self-check catches this — two lines currently report
FAIL (9,603 vs 7,775 expected, and 3,299 vs 3,019 expected).

The `clean_text()` fix on lines 131–132 was applied correctly and is working.

## Next steps

1. Fix the `corpus.py` bracket bug above; confirm all five checks say OK.
2. Fill the five TODOs in `preprocess.py`; confirm all six checks say OK.
3. Move on to `index.py` (inverted index).

## Environment

- Python 3.8.0, Windows. NLTK **3.9.1** installed.
- NLTK data downloaded: `punkt`, `punkt_tab`, `stopwords`, `wordnet`, `omw-1.4`.
  Note NLTK ≥3.9 needs `punkt_tab`, not just `punkt`. `preprocess.py` has an
  `ensure_nltk_data()` helper so teammates do not have to download by hand.
- `.sgm` files must be read as **latin-1**, not utf-8.
- pip is old (19.2.3); harmless, but expect a warning on install.
- Report should pin versions in a `requirements.txt` for reproducibility.
