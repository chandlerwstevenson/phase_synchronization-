"""Literature-audit harness.

Fixes the failure modes of our ad-hoc reviews: no query log (can't
tell what was searched), no synonym coverage (keyword misses papers
that use different vocabulary), no citation-graph walking (the one
method that is vocabulary-independent), and no claim-to-evidence
ledger (verdicts lived in chat transcripts).

Everything this tool does is recorded in ledger.json in this folder:
every query verbatim, every hit, every verdict. Re-running is
idempotent; the ledger only grows.

Commands (run from anywhere):
    python audit.py claim add "<claim-id>" "<claim text>"
    python audit.py expand "<claim-id>"        # print the query lattice
    python audit.py arxiv "<query>" [--n 20] [--claim ID]
    python audit.py s2 "<query>" [--n 20] [--claim ID]
    python audit.py s2-refs <arxivId-or-S2paperId> [--claim ID]
    python audit.py s2-cites <arxivId-or-S2paperId> [--claim ID]
    python audit.py note "<claim-id>" "<verdict/observation text>"
    python audit.py matrix                      # claim x evidence table
    python audit.py log [claim-id]              # what has been searched

The synonym lattice lives in synonyms.json (edit freely); expand
crosses each concept group in a claim's `concepts` list to generate
the query family, so coverage is systematic rather than remembered.
"""

from __future__ import annotations

import itertools
import json
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
LEDGER = HERE / "ledger.json"
SYNONYMS = HERE / "synonyms.json"

S2_FIELDS = "title,year,abstract,externalIds,citationCount,venue"
ATOM = "{http://www.w3.org/2005/Atom}"


def _load(path: Path, default):
    if path.exists():
        return json.loads(path.read_text())
    return default


def _save(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=1))


def _ledger():
    return _load(LEDGER, {"claims": {}, "queries": [], "notes": []})


def _http(url: str, tries: int = 5) -> str:
    delay = 2.0
    for attempt in range(tries):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "lit-audit/1.0"}
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read().decode("utf-8", "replace")
        except Exception as error:  # 429s and transient network
            if attempt == tries - 1:
                raise
            time.sleep(delay)
            delay *= 2
    return ""


def _record_query(source, query, hits, claim):
    ledger = _ledger()
    ledger["queries"].append(
        {
            "time": time.strftime("%Y-%m-%d %H:%M"),
            "source": source,
            "query": query,
            "claim": claim,
            "hits": hits,
        }
    )
    _save(LEDGER, ledger)


def arxiv_search(query: str, n: int = 20, claim: str | None = None):
    url = (
        "https://export.arxiv.org/api/query?search_query="
        + urllib.parse.quote(query)
        + f"&max_results={n}&sortBy=relevance"
    )
    root = ET.fromstring(_http(url))
    hits = []
    for entry in root.findall(f"{ATOM}entry"):
        arxiv_id = entry.findtext(f"{ATOM}id", "").split("/abs/")[-1]
        hits.append(
            {
                "id": arxiv_id,
                "title": " ".join(
                    entry.findtext(f"{ATOM}title", "").split()
                ),
                "year": entry.findtext(f"{ATOM}published", "")[:4],
                "abstract": " ".join(
                    entry.findtext(f"{ATOM}summary", "").split()
                )[:600],
            }
        )
    _record_query("arxiv", query, hits, claim)
    return hits


def s2_search(query: str, n: int = 20, claim: str | None = None):
    url = (
        "https://api.semanticscholar.org/graph/v1/paper/search?query="
        + urllib.parse.quote(query)
        + f"&limit={n}&fields={S2_FIELDS}"
    )
    data = json.loads(_http(url))
    hits = [
        {
            "id": (item.get("externalIds") or {}).get("ArXiv")
            or item.get("paperId"),
            "title": item.get("title"),
            "year": item.get("year"),
            "venue": item.get("venue"),
            "cites": item.get("citationCount"),
            "abstract": (item.get("abstract") or "")[:600],
        }
        for item in data.get("data", [])
    ]
    _record_query("s2", query, hits, claim)
    return hits


def s2_graph(paper: str, direction: str, claim: str | None = None):
    key = f"arXiv:{paper}" if "." in paper and "/" not in paper else paper
    field = "references" if direction == "refs" else "citations"
    url = (
        f"https://api.semanticscholar.org/graph/v1/paper/{key}/{field}"
        f"?limit=100&fields={S2_FIELDS}"
    )
    data = json.loads(_http(url))
    hits = []
    for item in data.get("data", []):
        paper_data = item.get("citedPaper") or item.get("citingPaper") or {}
        hits.append(
            {
                "id": (paper_data.get("externalIds") or {}).get("ArXiv")
                or paper_data.get("paperId"),
                "title": paper_data.get("title"),
                "year": paper_data.get("year"),
                "venue": paper_data.get("venue"),
                "cites": paper_data.get("citationCount"),
                "abstract": (paper_data.get("abstract") or "")[:400],
            }
        )
    _record_query(f"s2-{direction}:{paper}", paper, hits, claim)
    return hits


def expand(claim_id: str):
    ledger = _ledger()
    claim = ledger["claims"].get(claim_id)
    if not claim or "concepts" not in claim:
        print("claim missing or has no `concepts` list "
              "(edit ledger.json to add one)")
        return []
    groups = []
    synonyms = _load(SYNONYMS, {})
    for concept in claim["concepts"]:
        groups.append(synonyms.get(concept, [concept]))
    queries = []
    for combo in itertools.product(*groups):
        queries.append(" ".join(f'"{term}"' for term in combo))
    return queries


def main() -> None:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return
    command = args[0]

    def opt(flag, default=None):
        if flag in args:
            return args[args.index(flag) + 1]
        return default

    if command == "claim" and args[1] == "add":
        ledger = _ledger()
        ledger["claims"][args[2]] = {
            "text": args[3],
            "status": "open",
            "concepts": [],
        }
        _save(LEDGER, ledger)
        print(f"added claim {args[2]} (add `concepts` in ledger.json "
              "to enable expand)")
    elif command == "expand":
        for query in expand(args[1]):
            print(query)
    elif command in ("arxiv", "s2"):
        n = int(opt("--n", "20"))
        fn = arxiv_search if command == "arxiv" else s2_search
        for hit in fn(args[1], n, opt("--claim")):
            print(f"[{hit.get('year')}] {hit.get('id')}  "
                  f"{hit.get('title')}")
    elif command in ("s2-refs", "s2-cites"):
        for hit in s2_graph(
            args[1], command.split("-")[1], opt("--claim")
        ):
            print(f"[{hit.get('year')}] cites={hit.get('cites')} "
                  f"{hit.get('id')}  {hit.get('title')}")
    elif command == "note":
        ledger = _ledger()
        ledger["notes"].append(
            {
                "time": time.strftime("%Y-%m-%d %H:%M"),
                "claim": args[1],
                "text": args[2],
            }
        )
        _save(LEDGER, ledger)
        print("noted")
    elif command == "matrix":
        ledger = _ledger()
        for claim_id, claim in ledger["claims"].items():
            notes = [
                note for note in ledger["notes"]
                if note["claim"] == claim_id
            ]
            queries = [
                query for query in ledger["queries"]
                if query.get("claim") == claim_id
            ]
            print(f"\n== {claim_id} [{claim['status']}]: "
                  f"{claim['text'][:100]}")
            print(f"   queries run: {len(queries)}, notes: {len(notes)}")
            for note in notes:
                print(f"   - {note['text'][:200]}")
    elif command == "log":
        ledger = _ledger()
        for query in ledger["queries"]:
            if len(args) > 1 and query.get("claim") != args[1]:
                continue
            print(f"{query['time']} [{query['source']}] "
                  f"{query['query'][:90]}  -> {len(query['hits'])} hits")
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
