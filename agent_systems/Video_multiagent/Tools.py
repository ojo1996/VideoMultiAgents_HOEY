from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import os, time

@dataclass
class ToolResult:
    name: str
    args: Dict[str, Any]
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0
    artifacts: Optional[List[str]] = None
    latency_ms: Optional[int] = None

class FrameSampler:
    """Stub: pretend to sample N frames uniformly (no heavy deps)."""
    def __init__(self, num_frames: int = 8):
        self.num_frames = num_frames
        self.name = "frame_sampler"

    def __call__(self, video_path: str) -> ToolResult:
        t0 = time.time()
        if not os.path.exists(video_path):
            return ToolResult(
                name=self.name,
                args={"video_path": video_path, "num_frames": self.num_frames},
                stdout="",
                stderr=f"Video not found: {video_path}",
                returncode=2,
                artifacts=[],
                latency_ms=int((time.time() - t0) * 1000),
            )
        indices = list(range(self.num_frames))
        return ToolResult(
            name=self.name,
            args={"video_path": video_path, "num_frames": self.num_frames},
            stdout=f"sampled_indices={indices}",
            stderr="",
            returncode=0,
            artifacts=[f"{video_path}#frames={self.num_frames}"],
            latency_ms=int((time.time() - t0) * 1000),
        )

class OCRTool:
    """Stub: placeholder OCR."""
    def __init__(self):
        self.name = "ocr"

    def __call__(self, frames_info: str) -> ToolResult:
        t0 = time.time()
        text = f"ocr_text=<'no real ocr'; frames_info='{frames_info[:80]}...'>"
        return ToolResult(
            name=self.name,
            args={"frames_info": frames_info},
            stdout=text,
            stderr="",
            returncode=0,
            artifacts=[],
            latency_ms=int((time.time() - t0) * 1000),
        )

class TemporalReasoner:
    """Stub: simple heuristic (replace with LLM later)."""
    def __init__(self):
        self.name = "temporal_reasoner"

    def __call__(self, question: str, evidence: Dict[str, str]) -> ToolResult:
        t0 = time.time()
        answer = f"answer_stub: frames={evidence.get('frames','')} | ocr={evidence.get('ocr','')}"
        return ToolResult(
            name=self.name,
            args={"question": question, "evidence": evidence},
            stdout=answer,
            stderr="",
            returncode=0,
            artifacts=[],
            latency_ms=int((time.time() - t0) * 1000),
        )
