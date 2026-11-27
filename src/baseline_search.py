"""Run the Milestone 2 Boolean search entry point."""
from search import main as run_search
from search import boolean_search as search
from search import boolean_search_terms as search_terms


def main(argv=None):
    return run_search(argv, boolean=True)


if __name__ == "__main__":
    main()
