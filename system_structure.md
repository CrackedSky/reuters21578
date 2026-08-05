Six modules
The nice thing is that the assignment's required components map almost one-to-one onto files, so your code structure and your report structure end up matching.

1. corpus.py — get the documents out of the files

Reads the .sgm files and pulls out, for each article, its ID, title, and body text. It also reads the LEWISSPLIT and TOPICS attributes so you can pick which set of documents you want. Output: a list of documents, each roughly {id, text, categories}.

2. preprocess.py — turn text into clean words

One function that takes a string and returns a list of words. Inside it: lowercase, tokenise with NLTK, drop punctuation, drop stopwords, stem. This same function is used on documents and on queries — that's important, because both sides have to be processed identically or they won't match.

3. index.py — build the lookup table

Builds an inverted index: a dictionary mapping each word to the list of documents containing it, plus how many times. So {"oil": {doc12: 3, doc85: 1}, ...}. It's "inverted" because instead of document-to-words, it's word-to-documents — which is what makes search fast, since you only look at documents that actually contain a query word instead of scanning all 3,000.

4. search.py — score and rank

Takes a preprocessed query, looks up each word in the index, gives every candidate document a relevance score, and sorts. Start with TF-IDF: a word counts for more if it appears often in this document, and counts for less if it appears in lots of documents (common words like "said" shouldn't drive the ranking). Output: a ranked list of (doc_id, score).

5. evaluate.py — measure how good it is

Runs a batch of test queries, compares your ranked results against the known-correct answers, and prints scores like precision and recall. This is the module that gives you something to write in the evaluation section.

6. cli.py — the command line front end

Argparse. Ties it together: build an index, or run a search and print results.

How they connect
corpus.py loads documents, hands them to preprocess.py for cleaning, and the cleaned words go into index.py to build the index. Then at search time, a user query goes through the same preprocess.py, into search.py, which consults the index and returns ranked results. evaluate.py just automates that last path over many queries at once.

Only cli.py imports everything; the other five don't need to know about each other beyond that chain.