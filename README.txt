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
the new generation after all index files have been built successfully.
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
Use double quotes for an exact sequence of stemmed words in the destination
page's text.
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
heading, and bold text. Cosine divides by both vector lengths. A soft penalty
applies to pages under 85 tokens. Destination-text results come first, with
greater query term coverage ranked higher. The combined score breaks ties,
followed by numeric document ID.

Additional ranking signals

PageRank is computed offline from the indexed link graph and added to ranked
search. Use --pagerank-weight 0 to disable it.

HITS uses incoming and outgoing links around the top 40 query matches. The
neighborhood adds up to 20 neighbors per direction for each match and stops
at 300 pages. Hub and authority scores start at one, alternate updates,
and are normalized to sum to one for up to ten rounds. The ranker adds a
small boost to eligible results, with more weight on authority. Use
--hits-weight 0 to disable it.

Word positions support exact phrases and reward shorter matching windows in
unquoted queries. Use --proximity-weight 0 to disable the window boost.

Exact and near-duplicate content is grouped during indexing. Search shows
one result per content group and per URL. Use --show-duplicates to show
matching copies at separate URLs.

Incoming anchor words are indexed for their destinations. A word contributes
one vote per linking page and destination. Generic labels such as "click
here" are ignored. Anchor-only hits follow destination-text matches. Use
--anchor-weight 0 to disable this signal. Rebuild the index to generate the
offline data for these features.

Submission and demo

The submission ZIP contains the source programs, requirements.txt, README.txt,
and TEST.txt. TEST.txt records the evaluation queries. Use the supplied
Developer corpus to rebuild the index locally. In the demo, show the full
build's spill count, run a required query, and explain
where search reads postings and how the ranking weights are calculated.
