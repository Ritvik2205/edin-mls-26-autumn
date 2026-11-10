"""ASRAdapter that spawns the variant worker (extensions/fast_asr/worker.py) instead of
rag/src/asr_worker.py. Same JSON-lines protocol; transcribe/close are inherited unchanged."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

RAG = Path(__file__).resolve().parents[2]
if str(RAG) not in sys.path:
    sys.path.insert(0, str(RAG))

from src.asr_adapter import ASRAdapter  # noqa: E402

WORKER = Path(__file__).with_name("worker.py")


class FastASRAdapter(ASRAdapter):
    def __init__(self, variant="v2", margin_s=None, bucket_s=None, worker_path=None, **kw):
        if variant not in ("v1", "v2", "v3"):
            raise ValueError(f"unknown variant {variant}")
        if variant == "v3" and (margin_s is None) == (bucket_s is None):
            raise ValueError("v3 needs exactly one of margin_s or bucket_s")
        if variant != "v3" and (margin_s is not None or bucket_s is not None):
            raise ValueError("margin_s/bucket_s only apply to v3")
        super().__init__(**kw)
        self.variant, self.margin_s, self.bucket_s = variant, margin_s, bucket_s
        self.worker_path = Path(worker_path) if worker_path else WORKER

    def worker_args(self) -> list[str]:
        args = ["--asr-root", str(self.asr_root), "--cuda", self.cuda_groups,
                "--max-new-tokens", str(self.max_new_tokens), "--variant", self.variant]
        if self.margin_s is not None:
            args += ["--margin", f"{self.margin_s:g}"]
        if self.bucket_s is not None:
            args += ["--bucket", f"{self.bucket_s:g}"]
        return args

    def prepare(self):
        if self.process is not None:
            if self.process.poll() is not None:
                raise RuntimeError("ASR worker exited; see its error output")
            return
        env = os.environ.copy()
        env["HF_HUB_OFFLINE"] = "1"
        self.process = subprocess.Popen(
            [self.python, "-B", "-u", str(self.worker_path), *self.worker_args()],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None,
            text=True, encoding="utf-8", env=env,
        )
        try:
            if self._read_response().get("ready") is not True:
                raise RuntimeError("ASR worker did not report readiness")
        except BaseException:
            self.close()
            raise
