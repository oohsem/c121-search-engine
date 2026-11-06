from parser import parse_html
from indexer import DOCS_PER_PARTIAL, build_index, save_partial_index, merge_partial_indexes

DATASET_DIR = Path('dataset')
INDEX_DIR = Path('index')

def main():
    dataset = DATASET_DIR
    max_docs = None

    