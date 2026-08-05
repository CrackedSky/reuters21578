"""
corpus.py
=========
Step 1 of the IR pipeline: read the Reuters-21578 .sgm files and turn them
into a clean list of Document objects.

Note there is no NLTK in this file, and that is deliberate. This module is
pure "data loading": get bytes off disk, find the articles, pull out the
fields. All the actual NLP (tokenising, stopwords, stemming) happens later
in preprocess.py. Keeping the two apart means a parsing bug can never be
confused with an NLP bug.

Fill in the TODO blanks, then run this file directly to check your work:

    python corpus.py
"""

import os
import re
import glob
from dataclasses import dataclass
from typing import List

# The .sgm files live in the same folder as this script.
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# These files are NOT utf-8. They were produced in 1987/1996 and contain
# Western European bytes that utf-8 would crash on. latin-1 decodes any
# byte without error, so it is the safe choice for old corpora like this.
ENCODING = "latin-1"


# --------------------------------------------------------------------------
# The data structure we produce
# --------------------------------------------------------------------------

@dataclass
class Document:
    """One Reuters news article."""
    doc_id: int          # the NEWID attribute, unique across the collection
    title: str
    body: str
    categories: List[str]  # the TOPICS labels, e.g. ["grain", "wheat"]
    split: str           # LEWISSPLIT: "TRAIN", "TEST" or "NOT-USED"
    topics_flag: str     # TOPICS attribute: "YES", "NO" or "BYPASS"

    @property
    def text(self) -> str:
        """Title + body: the ONLY thing we are allowed to index and search.

        Never include `categories` here. Those labels are the answer key we
        grade against later in evaluate.py. If they leak into the searchable
        text, the system looks near-perfect and the score means nothing.
        """
        return self.title + " " + self.body


# --------------------------------------------------------------------------
# Regular expressions
#
# Two flags/tricks matter a lot here:
#   re.S  (DOTALL)  -> makes "." also match newlines. Article bodies run over
#                      many lines, so without this we would stop at line one.
#   .*?   (lazy)    -> match as FEW characters as possible, so we stop at the
#                      FIRST closing tag. Greedy ".*" would swallow the whole
#                      file up to the LAST closing tag.
# --------------------------------------------------------------------------

# Given: finds each article. group(1) = the attributes inside the opening
# tag, group(2) = everything between <REUTERS ...> and </REUTERS>.
DOC_RE = re.compile(r"<REUTERS(.*?)>(.*?)</REUTERS>", re.S)

# Given: pulls KEY="value" pairs out of that attribute text.
ATTR_RE = re.compile(r'(\w+)="([^"]*)"')

# Given, as your worked example: grabs the whole <TOPICS>...</TOPICS> block.
TOPICS_RE = re.compile(r"<TOPICS>(.*?)</TOPICS>", re.S)

# TODO(1): Write the same kind of pattern for <TITLE> and <BODY>.
#          Copy the shape of TOPICS_RE above and change the tag name.
TITLE_RE = re.compile(r"<TITLE>(.*?)</TITLE>", re.S)  # <-- your code here
BODY_RE = re.compile(r"<BODY>(.*?)</BODY>", re.S)   # <-- your code here

# TODO(2): Inside a TOPICS block, each label is wrapped in its own <D> tag,
#          like: <TOPICS><D>grain</D><D>wheat</D></TOPICS>
#          Write a pattern that matches ONE <D>...</D> pair. We will use
#          .findall() with it, which returns every match as a list.
D_RE = re.compile(r"<D>(.*?)</D>", re.S)  # <-- your code here


# --------------------------------------------------------------------------
# Small helpers (given -- these are fiddly but not conceptually interesting)
# --------------------------------------------------------------------------

def _extract(pattern, text: str, default: str = "") -> str:
    """Run `pattern` on `text` and return group(1), or `default` if no match.

    Why not just call .group(1) directly? Because real data is messy: about
    290 articles have a <TITLE> but no <BODY> at all. Calling .group(1) on a
    failed match raises AttributeError, so we check for None first.
    """
    match = pattern.search(text)
    return match.group(1) if match else default


def clean_text(text: str) -> str:
    """Tidy up SGML artefacts and whitespace."""
    text = re.sub(r"&#\d+;", " ", text)   # control chars, e.g. &#3;
    text = text.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
    return re.sub(r"\s+", " ", text).strip()   # collapse runs of whitespace


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------

def parse_file(path: str) -> List[Document]:
    """Parse ONE .sgm file into a list of Document objects."""
    documents = []

    with open(path, encoding=ENCODING) as f:
        raw = f.read()

    for match in DOC_RE.finditer(raw):
        attrs = dict(ATTR_RE.findall(match.group(1)))
        inner = match.group(2)

        # TODO(3): Pull the title and body out of `inner`.
        #          Use the _extract() helper with the patterns from TODO(1),
        #          then pass the result through clean_text().
        title = clean_text(_extract(TITLE_RE, inner))  # <-- your code here
        body = clean_text(_extract(BODY_RE, inner))  # <-- your code here

        # TODO(4): Build the list of category labels.
        #          Step a: use _extract() with TOPICS_RE to get the block.
        #          Step b: use D_RE.findall() on that block to get the labels.
        #          An article with no topics should give an empty list [].
        topics = _extract(TOPICS_RE, inner)
        categories = D_RE.findall(topics)  # <-- your code here

        documents.append(Document(
            doc_id=int(attrs["NEWID"]),
            title=title,
            body=body,
            categories=categories,
            split=attrs.get("LEWISSPLIT", ""),
            topics_flag=attrs.get("TOPICS", ""),
        ))

    return documents


def load_documents(split: str = "all") -> List[Document]:
    """Load the whole collection, optionally keeping only one ModApte split.

    `split` is one of: "all", "train", "test".
    """
    documents = []
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "reut2-*.sgm"))):
        documents.extend(parse_file(path))

    if split == "all":
        return documents

    # TODO(5): Apply the ModApte rule from README.txt section VIII.B.
    #
    #   training set = LEWISSPLIT is "TRAIN" AND TOPICS attribute is "YES"
    #   test set     = LEWISSPLIT is "TEST"  AND TOPICS attribute is "YES"
    #
    # Careful: `topics_flag` (the attribute) is NOT the same as whether
    # `categories` is non-empty. The README is explicit that a TOPICS="YES"
    # article can still have zero labels. Filter on the attribute only.
    #
    # Hint: a list comprehension reads nicely here, e.g.
    #     return [d for d in documents if <condition>]
    wanted_split = "TRAIN" if split == "train" else "TEST"
    return [d for d in documents if (d.split==wanted_split) and (d.topics_flag=="YES")]  # <-- your code here


# --------------------------------------------------------------------------
# Self-check: run `python corpus.py` to see whether your blanks are right.
# The expected numbers come from README.txt section VIII.
# --------------------------------------------------------------------------

if __name__ == "__main__":
    all_docs = load_documents("all")
    train = load_documents("train")
    test = load_documents("test")

    checks = [
        ("total articles", len(all_docs), 21578),
        ("ModApte train", len(train), 9603),
        ("ModApte test", len(test), 3299),
        ("train docs with >=1 category", sum(1 for d in train if d.categories), 7775),
        ("test docs with >=1 category", sum(1 for d in test if d.categories), 3019),
    ]

    print()
    for name, got, expected in checks:
        status = "OK  " if got == expected else "FAIL"
        print("[%s] %-30s got %6d, expected %6d" % (status, name, got, expected))

    # Eyeball one article to confirm the text actually looks like English.
    if train:
        sample = train[0]
        print("\n--- sample document ---")
        print("id:         ", sample.doc_id)
        print("categories: ", sample.categories)
        print("title:      ", sample.title)
        print("body:       ", sample.body[:200], "...")
