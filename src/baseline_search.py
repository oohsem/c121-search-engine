"""Run the Milestone 2 Boolean search entry point."""
from search import main as run_search
from search import search as search


def main(argv=None):
    return run_search(argv, boolean=True)


if __name__ == "__main__":
    main()
