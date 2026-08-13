"""
cli.py
======
Step 6, the last one: the command-line front end that ties everything
together. This is the only module that imports all the others; corpus,
preprocess, index, search and evaluate stay independent of each other.

Four commands:

    python cli.py build
    python cli.py search "coffee exports"
    python cli.py interactive
    python cli.py eval

Useful flags:

    python cli.py build --split test
    python cli.py search "gold mining" --top 5
    python cli.py search "gold mining" --model bm25
    python cli.py eval --model both
    python cli.py eval --verbose
"""

import argparse
import sys

from corpus import load_documents
from index import build_index, save_index
from search import search, search_bm25, format_results, get_index
from evaluate import build_qrels, evaluate_all


# --------------------------------------------------------------------------
# Command handlers
# --------------------------------------------------------------------------

def cmd_build(args) -> int:
    """Build an index from scratch and cache it to disk."""

    print("Loading the %s split..." % args.split)

    documents = load_documents(args.split)

    print("Indexing %d documents..." % len(documents))

    index = build_index(documents)

    filename = "index_%s.pkl" % args.split

    save_index(index, filename)

    print(
        "Done. %d documents, %d distinct terms -> %s"
        % (index.num_docs, len(index.postings), filename)
    )

    return 0


# --------------------------------------------------------------------------


def cmd_search(args) -> int:
    """Run one query and print the ranked results."""

    # Load the index
    index = get_index(args.split)

    # Choose ranking model
    if args.model == "bm25":

        results = search_bm25(
            index,
            args.query,
            args.top,
            k1=args.k1,
            b=args.b
        )

    else:

        results = search(
            index,
            args.query,
            args.top
        )

    # Display
    print("Model: %s" % args.model.upper())
    print("Query: %r" % args.query)

    print(format_results(index, results))

    return 0


# --------------------------------------------------------------------------


def cmd_interactive(args) -> int:
    """Prompt for queries in a loop until the user quits."""

    index = get_index(args.split)

    print(
        "\nReuters IR system using %s."
        % args.model.upper()
    )

    print("Type a query, or 'quit' to exit.\n")

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

        # Choose ranking method
        if args.model == "bm25":

            results = search_bm25(
                index,
                query,
                args.top,
                k1=args.k1,
                b=args.b
            )

        else:

            results = search(
                index,
                query,
                args.top
            )

        print(format_results(index, results))
        print()

    print("Bye.")

    return 0


# --------------------------------------------------------------------------


def cmd_eval(args) -> int:
    """Run every category as a query and report evaluation metrics."""

    # Load documents
    documents = load_documents(args.split)

    # Build relevance judgments
    qrels = build_qrels(documents)

    # Load index
    index = get_index(args.split)

    # If "both" selected, evaluate both models
    if args.model == "both":
        models = ["tfidf", "bm25"]

    else:
        models = [args.model]

    print(
        "Evaluated %d categories."
        % len(qrels)
    )

    # Evaluate each selected model
    for model in models:

        results = evaluate_all(
            index,
            qrels,
            top_k=args.top,
            verbose=args.verbose,
            model=model,
            k1=args.k1,
            b=args.b
        )

        print("\n%s" % model.upper())

        print(
            "MAP  = %.4f"
            % results["map"]
        )

        print(
            "P@10 = %.4f"
            % results["p@10"]
        )

        print(
            "R@10 = %.4f"
            % results["r@10"]
        )

    return 0


# --------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        prog="cli.py",
        description=(
            "A TF-IDF/BM25 information retrieval system "
            "over Reuters-21578."
        )
    )

    sub = parser.add_subparsers(
        dest="command",
        metavar="COMMAND"
    )


    # ----------------------------------------------------------------------
    # BUILD
    # ----------------------------------------------------------------------

    p_build = sub.add_parser(
        "build",
        help="build and cache an index"
    )

    p_build.add_argument(
        "--split",
        default="train",
        choices=["train", "test"],
        help="which ModApte split to index (default: train)"
    )

    p_build.set_defaults(
        func=cmd_build
    )


    # ----------------------------------------------------------------------
    # SEARCH
    # ----------------------------------------------------------------------

    p_search = sub.add_parser(
        "search",
        help="search for a query"
    )

    # Positional query
    p_search.add_argument(
        "query",
        help="query string to search for"
    )

    # Number of results
    p_search.add_argument(
        "--top",
        type=int,
        default=10,
        help="number of results to show (default: 10)"
    )

    # Dataset split
    p_search.add_argument(
        "--split",
        default="train",
        choices=["train", "test"],
        help="which ModApte split to search (default: train)"
    )

    # Ranking model
    p_search.add_argument(
        "--model",
        default="tfidf",
        choices=["tfidf", "bm25"],
        help="ranking model: tfidf or bm25 (default: tfidf)"
    )

    # BM25 k1
    p_search.add_argument(
        "--k1",
        type=float,
        default=1.5,
        help="BM25 k1 parameter (default: 1.5)"
    )

    # BM25 b
    p_search.add_argument(
        "--b",
        type=float,
        default=0.75,
        help="BM25 b parameter (default: 0.75)"
    )

    p_search.set_defaults(
        func=cmd_search
    )


    # ----------------------------------------------------------------------
    # INTERACTIVE SEARCH
    # ----------------------------------------------------------------------

    p_int = sub.add_parser(
        "interactive",
        help="type queries in a loop"
    )

    p_int.add_argument(
        "--top",
        type=int,
        default=10
    )

    p_int.add_argument(
        "--split",
        default="train",
        choices=["train", "test"]
    )

    p_int.add_argument(
        "--model",
        default="tfidf",
        choices=["tfidf", "bm25"]
    )

    p_int.add_argument(
        "--k1",
        type=float,
        default=1.5
    )

    p_int.add_argument(
        "--b",
        type=float,
        default=0.75
    )

    p_int.set_defaults(
        func=cmd_interactive
    )


    # ----------------------------------------------------------------------
    # EVALUATION
    # ----------------------------------------------------------------------

    p_eval = sub.add_parser(
        "eval",
        help="run the full evaluation experiment"
    )

    p_eval.add_argument(
        "--split",
        default="train",
        choices=["train", "test"]
    )

    p_eval.add_argument(
        "--top",
        type=int,
        default=100,
        help="how many results to score per query (default: 100)"
    )

    p_eval.add_argument(
        "--verbose",
        action="store_true",
        help="print a line per category"
    )

    p_eval.add_argument(
        "--model",
        default="both",
        choices=["tfidf", "bm25", "both"],
        help="model to evaluate: tfidf, bm25, or both (default: both)"
    )

    p_eval.add_argument(
        "--k1",
        type=float,
        default=1.5,
        help="BM25 k1 parameter (default: 1.5)"
    )

    p_eval.add_argument(
        "--b",
        type=float,
        default=0.75,
        help="BM25 b parameter (default: 0.75)"
    )

    p_eval.set_defaults(
        func=cmd_eval
    )

    return parser


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main(argv=None) -> int:

    parser = build_parser()

    args = parser.parse_args(argv)

    # No command -> show help
    if not getattr(args, "func", None):

        parser.print_help()

        return 1

    return args.func(args)


# --------------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(main())