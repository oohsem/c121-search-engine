from flask import Flask, request, render_template_string
from ranking import Ranker
from search_index import IndexReader

app = Flask(__name__)

PAGE = """
<h1>Search Engine</h1>
<form><input name="q" value="{{ query }}" autofocus> <button>Search</button></form>
{% for url in urls %}
  <p><a href="{{ url }}">{{ url }}</a></p>
{% endfor %}
{% if query and not urls %}<p>No results.</p>{% endif %}
"""

@app.route("/")
def home():
    query = request.args.get("q", "").strip()
    urls = ranker.search(query, 10) if query else []
    return render_template_string(PAGE, query=query, urls=urls)

if __name__ == "__main__":
    with IndexReader() as index:
        ranker = Ranker(index)
        app.run(port=8000)