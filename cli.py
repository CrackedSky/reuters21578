"""
cli.py
======
Step 6, the last one: the command-line front end that ties everything
together. This is the only module that imports all the others; corpus,
preprocess, index, search and evaluate stay independent of each other.

Four commands:

    python cli.py build                        build and cache an index
    python cli.py search "coffee exports"      one-off search
    python cli.py interactive                  type queries in a loop
    python cli.py eval                         run the full experiment

Useful flags:

    python cli.py build  --split test
    python cli.py search "gold mining" --top 5
    python cli.py eval   --verbose

Fill in the TODO blanks, then try:

    python cli.py --help
    python cli.py search "cocoa buffer stock"
"""

import argparse
import sys

from corpus import load_documents
from index import build_index, save_index
from search import search, format_results, get_index
from evaluate import build_qrels, evaluate_all


# --------------------------------------------------------------------------
# Command handlers
# --------------------------------------------------------------------------

def cmd_build(args) -> int:
    """Build an index from scratch and cache it to disk. (Given, as your
    worked example of what a handler looks like.)"""
    print("Loading the %s split..." % args.split)
    documents = load_documents(args.split)
    print("Indexing %d documents (about 35 seconds)..." % len(documents))

    index = build_index(documents)
    filename = "index_%s.pkl" % args.split

    save_index(index, filename)
    print("Done. %d documents, %d distinct terms -> %s"
          % (index.num_docs, len(index.postings), filename))
    return 0


def cmd_search(args) -> int:
    """Run one query and print the ranked results.

    TODO(2): Three steps.
             a. index = get_index(args.split)
                get_index() loads the cached .pkl, or builds it the first
                time, so the user never has to remember to run `build`.
             b. results = search(index, args.query, args.top)
             c. print the query, then print format_results(index, results)

             format_results() already handles the empty case, so you do not
             need to special-case "no matches" yourself.
    """
    # <-- your code here
    return 0


def cmd_interactive(args) -> int:
    """Prompt for queries in a loop until the user quits. (Given.)"""
    index = get_index(args.split)
    print("\nReuters IR system. Type a query, or 'quit' to exit.\n")

    while True:
        try:
            query = input("query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            break
        print(format_results(index, search(index, query, args.top)))
        print()

    print("Bye.")
    return 0


def cmd_eval(args) -> int:
    """Run every category as a query and report the averaged metrics.

    TODO(3): Four steps.
             a. documents = load_documents(args.split)
             b. qrels = build_qrels(documents)
             c. index = get_index(args.split)
             d. results = evaluate_all(index, qrels, top_k=args.top,
                                       verbose=args.verbose)

             Then print MAP, P@10 and R@10 from the returned dict, plus how
             many categories were evaluated (len(qrels)), because a metric
             without its query count is not reproducible.
    """
    # <-- your code here
    return 0


# --------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cli.py",
        description="A TF-IDF information retrieval system over Reuters-21578.")

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # --- build (given, as the pattern to copy) ---------------------------
    p_build = sub.add_parser("build", help="build and cache an index")
    p_build.add_argument("--split", default="train", choices=["train", "test"],
                         help="which ModApte split to index (default: train)")
    p_build.set_defaults(func=cmd_build)

    # --- search ----------------------------------------------------------
    p_search = sub.add_parser("search", help="search for a query")
    # TODO(1): Give the search command its arguments. Copy the style above.
    #
    #    a. a POSITIONAL argument "query" (no dashes), so the user can write
    #       cli.py search "coffee exports brazil". Give it help text.
    #    b. an optional "--top", type=int, default=10, for how many results
    #       to show.
    #    c. an optional "--split", exactly like the one on p_build above.
    #    d. p_search.set_defaults(func=cmd_search)
    #
    # <-- your code here

    # --- interactive (given) ---------------------------------------------
    p_int = sub.add_parser("interactive", help="type queries in a loop")
    p_int.add_argument("--top", type=int, default=10)
    p_int.add_argument("--split", default="train", choices=["train", "test"])
    p_int.set_defaults(func=cmd_interactive)

    # --- eval (given) ----------------------------------------------------
    p_eval = sub.add_parser("eval", help="run the full evaluation experiment")
    p_eval.add_argument("--split", default="train", choices=["train", "test"])
    p_eval.add_argument("--top", type=int, default=100,
                        help="how many results to score per query (default: 100)")
    p_eval.add_argument("--verbose", action="store_true",
                        help="print a line per category")
    p_eval.set_defaults(func=cmd_eval)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # No command given -> show help rather than a bare traceback.
    if not getattr(args, "func", None):
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
