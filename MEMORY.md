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

## Modules (6) — all six now exist as skeletons

| # | File | Job | Status |
|---|---|---|---|
| 1 | `corpus.py` | Parse `.sgm` files into `Document` objects; filter by split | **Complete**, all 5 checks pass |
| 2 | `preprocess.py` | Raw text → clean list of terms (NLTK) | **Complete**, all 6 checks pass |
| 3 | `index.py` | Build inverted index (term → documents containing it) | TODO 3, 4 done; **1, 2, 5 open** |
| 4 | `search.py` | Score and rank documents with TF-IDF cosine | Skeleton, 5 TODOs open |
| 5 | `evaluate.py` | Precision/recall/MAP against category pseudo-qrels | Skeleton, 5 TODOs open |
| 6 | `cli.py` | argparse front end tying it together | Skeleton, 3 TODOs open |

Suggested split for two people: one takes `index.py` + `search.py` (the
ranking maths), the other takes `evaluate.py` + `cli.py` (metrics and front
end). They only meet at function signatures, which are already fixed by the
skeletons, so the two halves can be written in parallel without conflicts.

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

## Verified baseline (what "correct" looks like)

These come from a working reference implementation run against the training
index, and are baked into the self-checks. Anything filled in correctly should
reproduce them exactly.

- Index: N = 9,603 documents, vocabulary = 19,229 terms, ~35 s to build.
- `df('cocoa')` = 59, `idf('cocoa')` ≈ 2.212; `df('bank')` = 1,859, idf ≈ 0.713.
- Document 1 norm ≈ 27.395.
- Search for `"cocoa"`: top hit is document 10471 at score ≈ 0.4767.
- Toy corpus: query `"cat dog"` scores document 2 at exactly 1.0, because that
  document *is* `['cat', 'dog']` — same terms, same proportions, zero angle.
- Evaluation over all 115 training categories: **MAP = 0.3487**.
- Per category: cocoa AP = 0.945, coffee = 0.815, sugar = 0.701, gold = 0.299,
  ship = 0.208. Rare, distinctive topics do well; broad ones do badly.
- MAP over the 10 largest categories is only 0.179, versus 0.349 overall —
  the head/tail contrast is the most interesting result in the dataset and
  should be a figure in the report.

## Next steps

1. Finish `index.py` TODOs 1, 2 and 5.
2. Fill `search.py` (5 TODOs), then `evaluate.py` (5), then `cli.py` (3).
3. Once `cli.py eval` reproduces MAP = 0.3487, that is the baseline. Every
   later improvement gets compared against it — that comparison is the
   experimental evaluation section of the report.

Ideas already discussed for improvements, in rough order of effort: a
corpus-specific stopword list (NLTK's 198 words miss "throughout",
"although", "said"); building queries from `cat-descriptions_120396.txt`
instead of bare category codes; lemmatisation instead of stemming; then BM25,
query expansion and spelling correction if time allows.

## Environment

- Python 3.8.0, Windows. NLTK **3.9.1** installed.
- NLTK data downloaded: `punkt`, `punkt_tab`, `stopwords`, `wordnet`, `omw-1.4`.
  Note NLTK ≥3.9 needs `punkt_tab`, not just `punkt`. `preprocess.py` has an
  `ensure_nltk_data()` helper so teammates do not have to download by hand.
- `.sgm` files must be read as **latin-1**, not utf-8.
- pip is old (19.2.3); harmless, but expect a warning on install.
- Report should pin versions in a `requirements.txt` for reproducibility.
