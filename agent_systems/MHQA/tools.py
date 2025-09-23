from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Callable
import time, os, json, hashlib, math, urllib.parse, urllib.request


#
# Minimal, model-agnostic tool layer for an MHQA-style agent.
#
# Goals:
# - Provide a tiny set of pluggable tools (BM25, Dense, Merge, Reader)
# - Standardize tool specifications and their vector representations
# - Validate tool vectors so downstream pipelines can trust them
# - Keep logic deterministic and dependency-light (no ML models here)
#


@dataclass
class ToolResult:
    """Standard return type for all tools.

    - name: tool identifier (matches spec["id"])  
    - args: echo of the input args used to invoke the tool  
    - stdout: primary payload as JSON string for easy piping across steps  
    - stderr: error message, if any  
    - returncode: 0 success, non-zero indicates error  
    - latency_ms: simple wall clock timing to aid analysis
    """
    name: str
    args: Dict[str, Any]
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0
    latency_ms: Optional[int] = None


class ToolRegistry:
    """Lightweight registry that keeps both specs and callables.

    We keep specs alongside implementations to ensure that any exported
    vectors or metadata can be traced back to a stable, versioned spec.
    """

    def __init__(self):
        self._tools: Dict[str, Callable[..., ToolResult]] = {}
        self._specs: Dict[str, Dict[str, Any]] = {}

    def register(self, spec: Dict[str, Any], impl: Callable[..., ToolResult]):
        self._tools[spec["id"]] = impl
        self._specs[spec["id"]] = spec

    def get(self, tool_id: str) -> Callable[..., ToolResult]:
        return self._tools[tool_id]

    def specs(self) -> List[Dict[str, Any]]:
        return list(self._specs.values())


def _http_get_json(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Tiny helper for best-effort HTTP GET returning parsed JSON.

    We deliberately avoid adding heavy HTTP deps; stdlib is enough here.
    """
    qs = urllib.parse.urlencode(params)
    with urllib.request.urlopen(f"{url}?{qs}", timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


# ------------------------ Domain Tools ------------------------ #


class BM25Search:
    """Optional HTTP client to a BM25 retrieval service.

    If `BM25_API_URL` is unset, we return a minimal stub so trajectories
    still have realistic shapes for debugging/analysis.
    """

    spec = {
        "id": "bm25_search",
        "name": "BM25 Search",
        "version": "1.0",
        "inputs": {"q": "str", "k": "int"},
        "outputs": {"docs": "List[Dict]"},
        "description": "HTTP client to BM25 service; falls back to stub.",
    }

    def __init__(self):
        self.url = os.getenv("BM25_API_URL")

    def __call__(self, q: str, k: int = 5) -> ToolResult:
        t0 = time.time()
        try:
            if not self.url:
                docs = [{"title": "Stub BM25", "text": f"fallback for: {q}"}]
            else:
                docs = _http_get_json(self.url, {"q": q, "k": k}).get("docs", [])
            return ToolResult(
                "bm25_search",
                {"q": q, "k": k},
                json.dumps({"docs": docs}),
                latency_ms=int((time.time() - t0) * 1000),
            )
        except Exception as e:  # pragma: no cover - defensive
            return ToolResult("bm25_search", {"q": q, "k": k}, "", str(e), 2)


class DenseSearch:
    """Optional HTTP client to a dense retrieval service (e.g., BGE/Contriever)."""

    spec = {
        "id": "dense_search",
        "name": "Dense Search",
        "version": "1.0",
        "inputs": {"q": "str", "k": "int"},
        "outputs": {"docs": "List[Dict]"},
        "description": "HTTP client to dense retriever; falls back to stub.",
    }

    def __init__(self):
        self.url = os.getenv("DENSE_API_URL")

    def __call__(self, q: str, k: int = 5) -> ToolResult:
        t0 = time.time()
        try:
            if not self.url:
                docs = [{"title": "Stub Dense", "text": f"fallback for: {q}"}]
            else:
                docs = _http_get_json(self.url, {"q": q, "k": k}).get("docs", [])
            return ToolResult(
                "dense_search",
                {"q": q, "k": k},
                json.dumps({"docs": docs}),
                latency_ms=int((time.time() - t0) * 1000),
            )
        except Exception as e:  # pragma: no cover - defensive
            return ToolResult("dense_search", {"q": q, "k": k}, "", str(e), 2)


class HybridMerge:
    """Merge BM25 + Dense results with simple title+text dedupe and top-K cut."""

    spec = {
        "id": "hybrid_merge",
        "name": "Hybrid Merge",
        "version": "1.0",
        "inputs": {"bm25_json": "str", "dense_json": "str", "k": "int"},
        "outputs": {"docs": "List[Dict]"},
        "description": "Merge BM25 + Dense with simple dedupe, top-K.",
    }

    def __call__(self, bm25_json: str, dense_json: str, k: int = 10) -> ToolResult:
        import hashlib as hh
        t0 = time.time()
        try:
            b = json.loads(bm25_json).get("docs", [])
            d = json.loads(dense_json).get("docs", [])
            seen, merged = set(), []
            for doc in (b + d):
                key = hh.md5((doc.get("title", "") + "|" + doc.get("text", "")).encode()).hexdigest()
                if key not in seen:
                    merged.append(doc)
                    seen.add(key)
                if len(merged) >= k:
                    break
            return ToolResult(
                "hybrid_merge",
                {"k": k},
                json.dumps({"docs": merged}),
                latency_ms=int((time.time() - t0) * 1000),
            )
        except Exception as e:  # pragma: no cover - defensive
            return ToolResult("hybrid_merge", {"k": k}, "", str(e), 2)


class HeuristicReader:
    """Very light reader that picks a salient sentence or the first 12 words.

    This is a deterministic, zero-dependency placeholder so that the pipeline
    exhibits a full trajectory without requiring any ML inference.
    """

    spec = {
        "id": "heuristic_reader",
        "name": "Heuristic Reader",
        "version": "1.0",
        "inputs": {"q": "str", "merged_json": "str"},
        "outputs": {"answer": "str"},
        "description": "Pick salient sentence or first 12 words from top doc.",
    }

    def __call__(self, q: str, merged_json: str) -> ToolResult:
        import re
        t0 = time.time()
        try:
            docs = json.loads(merged_json).get("docs", [])
            text = " ".join([d.get("text", "") for d in docs[:1]])
            sents = re.split(r"(?<=[.!?])\s+", text)
            pick = next(
                (s.strip() for s in sents if re.search(r"[A-Z][a-z]{2,}", s) or re.search(r"\d", s)),
                None,
            )
            if not pick:
                pick = " ".join(text.split()[:12]).strip() or "N/A"
            return ToolResult(
                "heuristic_reader",
                {"q": q},
                json.dumps({"answer": pick}),
                latency_ms=int((time.time() - t0) * 1000),
            )
        except Exception as e:  # pragma: no cover - defensive
            return ToolResult("heuristic_reader", {"q": q}, "", str(e), 2)


# ------------------------ Vectors + Validation ------------------------ #


def _hash_float_0_1(s: str) -> float:
    """Map a string to a deterministic float in [0, 1]."""
    h = hashlib.md5(s.encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def text_hash_vector(text: str, dim: int = 128) -> List[float]:
    """Bag-of-hashed-words vectorizer.

    This is intentionally simple and deterministic. It provides a stable
    embedding-like vector for tasks and tool specs without requiring models.
    """
    tokens = [t for t in text.lower().split() if t]
    vec = [0.0] * dim
    for tok in tokens:
        idx = int(_hash_float_0_1(tok) * dim) % dim
        vec[idx] += 1.0
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def tool_to_vector(spec: Dict[str, Any], dim: int = 128) -> Dict[str, Any]:
    """Create a deterministic vector for a tool spec.

    We use the tool name/version plus I/O field names and description as the
    textual seed. A checksum of the spec is included for integrity checks.
    """
    desc = (
        f'{spec["name"]} {spec["version"]} '
        + " ".join(list(spec.get("inputs", {}).keys()) + list(spec.get("outputs", {}).keys()))
        + " "
        + spec.get("description", "")
    )
    vec = text_hash_vector(desc, dim)
    checksum = hashlib.md5(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    return {
        "tool_id": spec["id"],
        "dim": dim,
        "vector": vec,
        "metadata": {
            "name": spec.get("name"),
            "version": spec.get("version"),
            "tags": [],
            "checksum": checksum,
        },
    }


def validate_tool_vector(tv: Dict[str, Any]) -> List[str]:
    """Return a list of validation errors (empty list means valid)."""
    errs: List[str] = []
    if not isinstance(tv.get("tool_id"), str):
        errs.append("tool_id must be str")
    vec = tv.get("vector")
    if not isinstance(vec, list) or not vec:
        errs.append("vector must be non-empty list")
    if isinstance(vec, list) and any((not isinstance(x, (int, float)) or not math.isfinite(x)) for x in vec):
        errs.append("vector elements must be finite numbers")
    if tv.get("dim") != (len(vec) if isinstance(vec, list) else None):
        errs.append("dim must equal len(vector)")
    # Use a small epsilon to tolerate floating-point noise for near-zero norms
    if sum((x * x for x in (vec or []))) <= 1e-12:
        errs.append("vector norm must be > 0")
    meta = tv.get("metadata", {})
    if "checksum" not in meta:
        errs.append("metadata.checksum required")
    return errs


def register_default_tools() -> ToolRegistry:
    """Factory that returns a registry populated with default tools."""
    reg = ToolRegistry()
    reg.register(BM25Search.spec, BM25Search())
    reg.register(DenseSearch.spec, DenseSearch())
    reg.register(HybridMerge.spec, HybridMerge())
    reg.register(HeuristicReader.spec, HeuristicReader())
    return reg


