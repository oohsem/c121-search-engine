CS121 Project 3
Developer track, Milestone 3

Setup

Use Python 3.11 or newer. Extract the supplied Developer collection so the
project contains dataset/<domain>/<page>.json. Run these commands from the project directory.

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements.txt

Build the index

    python src/main.py

The indexer processes the full collection. It writes sorted partial indexes
to disk, clears the in-memory batch, and merges the partials. The default batch
budget is 100,000 postings. The builder flushes the current batch before
processing unusually large pages. It counts stems as a stream and extracts
text from oversized HTML without holding a full document tree. It publishes
the new generation after the merge and compact conversion both succeed.
The build prints indexed and skipped document counts, spill count, and the
path to milestone1_report.txt. An interrupted build leaves the last complete
generation available.

Run these commands for a quick build in a separate directory.

    python src/main.py --max-docs 500
    python src/search.py --index-dir smoke_index/final_index

Search

    python src/search.py

Try cristina lopes, machine learning, ACM, or master of software engineering.
Enter a blank line to quit. Results show ranked URLs and query time in
milliseconds. The timer includes retrieval, ranking, and result assembly
and stops before printing. Runs above 300 ms are marked.

Run the Milestone 2 conjunctive baseline with this command.

    python src/baseline_search.py

Use --keep-stopwords to keep stopwords in a ranked query.
Use --snippets to show short matching excerpts from the local corpus.
Use --dataset-dir when those source pages are outside the default dataset
directory. See python src/search.py --help for ranking and index options.

Index and ranking

The ranker loads document URLs and precomputed norms at startup and caches a
bounded set of lexicon buckets. Each query seeks directly to the relevant
ranges in compact postings on disk. To verify the binary postings and the
lexicon against the merged JSONL, run this command.

    python src/optimize.py --verify

Ranking follows Lecture 21's lnc.ltc cosine weighting. Log TF is 1 + log10(tf)
for positive counts. Query weights multiply it by log10(N/df), where N is
the number of indexed documents and df is the number containing the term.
The document vector uses log TF and a 0.35 log-weight bonus for title,
heading, and bold text. Cosine divides by both vector lengths. Results
matching more query terms come first; cosine and numeric document ID order
results within each tier. Only the best result for each exact URL is shown.
A soft penalty applies to pages under 85 tokens.

Submission and demo

The submission ZIP contains the source programs, requirements.txt, README.txt,
and TEST.txt. Use the supplied Developer corpus to rebuild the index locally. In the
demo, show the full build's spill count, run a required query, and explain
where search reads postings and how the ranking weights are calculated.
