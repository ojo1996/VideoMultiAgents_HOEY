from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import os, time, json, urllib.parse, urllib.request

@dataclass
class ToolResult:
    name: str
    args: Dict[str, Any]
    stdout: str = ""     # main payload (JSON string)
    stderr: str = ""
    returncode: int = 0
    latency_ms: Optional[int] = None

def _http_get_json(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    qs = urllib.parse.urlencode(params)
    with urllib.request.urlopen(f"{url}?{qs}", timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))

class BM25SearchTool:
    """
    Optional HTTP client to a BM25 retrieval service.
    Set env BM25_API_URL to something like: http://localhost:8001/search
    Expected JSON: {"docs":[{"title": "...", "text":"..."}, ...]}
    """
    def __init__(self):
        self.name = "bm25_search"
        self.url = os.environ.get("BM25_API_URL")  # None means stub

    def __call__(self, question: str, k: int = 5) -> ToolResult:
        t0 = time.time()
        try:
            if not self.url:
                # Stub fallback
                docs = [{"title":"Stub BM25", "text":f"No BM25 service. Fallback for: {question}"}]
                return ToolResult(self.name, {"q": question, "k": k},
                                  stdout=json.dumps({"docs": docs}),
                                  latency_ms=int((time.time()-t0)*1000))
            data = _http_get_json(self.url, {"q": question, "k": k})
            return ToolResult(self.name, {"q": question, "k": k},
                              stdout=json.dumps(data),
                              latency_ms=int((time.time()-t0)*1000))
        except Exception as e:
            return ToolResult(self.name, {"q": question, "k": k}, stderr=str(e), returncode=2)

class DenseSearchTool:
    """
    Optional HTTP client to a dense retrieval API (e.g., Contriever/BGE).
    Set env DENSE_API_URL, e.g., http://localhost:8002/search
    Expected JSON: {"docs":[{"title": "...", "text":"..."}, ...]}
    """
    def __init__(self):
        self.name = "dense_search"
        self.url = os.environ.get("DENSE_API_URL")

    def __call__(self, question: str, k: int = 5) -> ToolResult:
        t0 = time.time()
        try:
            if not self.url:
                docs = [{"title":"Stub Dense", "text":f"No dense service. Fallback for: {question}"}]
                return ToolResult(self.name, {"q": question, "k": k},
                                  stdout=json.dumps({"docs": docs}),
                                  latency_ms=int((time.time()-t0)*1000))
            data = _http_get_json(self.url, {"q": question, "k": k})
            return ToolResult(self.name, {"q": question, "k": k},
                              stdout=json.dumps(data),
                              latency_ms=int((time.time()-t0)*1000))
        except Exception as e:
            return ToolResult(self.name, {"q": question, "k": k}, stderr=str(e), returncode=2)

class HybridMergeTool:
    """Merge BM25 + Dense results with simple title+text dedupe and top-K cut."""
    def __init__(self): self.name = "hybrid_merge"

    def __call__(self, bm25_json: str, dense_json: str, k: int = 10) -> ToolResult:
        import hashlib
        t0 = time.time()
        try:
            b = json.loads(bm25_json).get("docs", [])
            d = json.loads(dense_json).get("docs", [])
            seen, merged = set(), []
            for doc in (b + d):
                key = hashlib.md5((doc.get("title","") + "|" + doc.get("text","")).encode("utf-8")).hexdigest()
                if key not in seen:
                    merged.append(doc)
                    seen.add(key)
                if len(merged) >= k:
                    break
            return ToolResult(self.name, {"k": k},
                              stdout=json.dumps({"docs": merged}),
                              latency_ms=int((time.time()-t0)*1000))
        except Exception as e:
            return ToolResult(self.name, {"k": k}, stderr=str(e), returncode=2)

class HeuristicReader:
    """
    Very light reader: pick the first sentence containing a named entity-ish token
    or a number; otherwise take the first ~12 words from the top doc. Pure stub,
    but deterministic and useful for trajectory shape.
    """
    def __init__(self): self.name = "heuristic_reader"

    def __call__(self, question: str, merged_json: str) -> ToolResult:
        import re
        t0 = time.time()
        try:
            docs = json.loads(merged_json).get("docs", [])
            text = " ".join([d.get("text","") for d in docs[:1]])  # top doc only
            # crude sentence split
            sents = re.split(r"(?<=[.!?])\s+", text)
            pick = None
            for s in sents:
                if re.search(r"[A-Z][a-z]{2,}", s) or re.search(r"\d", s):
                    pick = s.strip()
                    break
            if not pick:
                pick = " ".join(text.split()[:12]).strip() or "N/A"
            return ToolResult(self.name, {"q": question}, stdout=json.dumps({"answer": pick}),
                              latency_ms=int((time.time()-t0)*1000))
        except Exception as e:
            return ToolResult(self.name, {"q": question}, stderr=str(e), returncode=2)