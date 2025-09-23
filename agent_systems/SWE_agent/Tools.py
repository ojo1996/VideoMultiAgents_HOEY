from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import subprocess, shlex, os, time

@dataclass
class ToolResult:
    name: str
    args: Dict[str, Any]
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0
    artifacts: Optional[List[str]] = None
    latency_ms: Optional[int] = None

class BashTool:
    """Run a single shell command via subprocess."""
    def __init__(self, cwd: Optional[str] = None, timeout: int = 60):
        self.name = "bash"
        self.cwd = cwd
        self.timeout = timeout

    def __call__(self, cmd: str) -> ToolResult:
        t0 = time.time()
        try:
            parts = shlex.split(cmd)
            p = subprocess.run(
                parts,
                cwd=self.cwd,
                capture_output=True,
                timeout=self.timeout,
                text=True
            )
            return ToolResult(
                name=self.name,
                args={"cmd": cmd},
                stdout=p.stdout.strip(),
                stderr=p.stderr.strip(),
                returncode=p.returncode,
                artifacts=[],
                latency_ms=int((time.time() - t0) * 1000),
            )
        except Exception as e:
            return ToolResult(
                name=self.name,
                args={"cmd": cmd},
                stdout="",
                stderr=str(e),
                returncode=2,
                artifacts=[],
                latency_ms=int((time.time() - t0) * 1000),
            )

class FileEditTool:
    """Create or overwrite a file with given content."""
    def __init__(self):
        self.name = "file_edit"

    def write(self, path: str, content: str) -> ToolResult:
        t0 = time.time()
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return ToolResult(
                name=self.name,
                args={"op": "write", "path": path, "nchars": len(content)},
                stdout=f"wrote {len(content)} chars to {path}",
                stderr="",
                returncode=0,
                artifacts=[path],
                latency_ms=int((time.time() - t0) * 1000),
            )
        except Exception as e:
            return ToolResult(
                name=self.name,
                args={"op": "write", "path": path},
                stdout="",
                stderr=str(e),
                returncode=2,
                artifacts=[],
                latency_ms=int((time.time() - t0) * 1000),
            )