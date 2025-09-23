from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import time, math

@dataclass
class ToolResult:
    name: str
    args: Dict[str, Any]
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0
    artifacts: Optional[List[str]] = None
    latency_ms: Optional[int] = None

class AddTool:
    def __init__(self): self.name = "add"
    def __call__(self, a: float, b: float) -> ToolResult:
        t0 = time.time()
        try:
            val = float(a) + float(b)
            return ToolResult(self.name, {"a": a, "b": b}, stdout=str(val),
                              latency_ms=int((time.time()-t0)*1000))
        except Exception as e:
            return ToolResult(self.name, {"a": a, "b": b}, stderr=str(e), returncode=2)

class MultiplyTool:
    def __init__(self): self.name = "multiply"
    def __call__(self, a: float, b: float) -> ToolResult:
        t0 = time.time()
        try:
            val = float(a) * float(b)
            return ToolResult(self.name, {"a": a, "b": b}, stdout=str(val),
                              latency_ms=int((time.time()-t0)*1000))
        except Exception as e:
            return ToolResult(self.name, {"a": a, "b": b}, stderr=str(e), returncode=2)

class SquareTool:
    def __init__(self): self.name = "square"
    def __call__(self, x: float) -> ToolResult:
        t0 = time.time()
        try:
            val = float(x) * float(x)
            return ToolResult(self.name, {"x": x}, stdout=str(val),
                              latency_ms=int((time.time()-t0)*1000))
        except Exception as e:
            return ToolResult(self.name, {"x": x}, stderr=str(e), returncode=2)

def eval_expression(expr: str) -> ToolResult:
    """Safe-ish eval for +,*,**,(),digits,spaces,^ (as **)."""
    import re, time
    t0 = time.time()
    if not re.fullmatch(r"[0-9\.\s\+\*\(\)\^xX\-]+", expr):
        return ToolResult("eval_expression", {"expr": expr}, stderr="unsupported chars", returncode=2)
    safe = expr.replace("^", "**")
    try:
        val = eval(safe, {"__builtins__": {}}, {"math": math})
        return ToolResult("eval_expression", {"expr": expr}, stdout=str(val),
                          latency_ms=int((time.time()-t0)*1000))
    except Exception as e:
        return ToolResult("eval_expression", {"expr": expr}, stderr=str(e), returncode=2)