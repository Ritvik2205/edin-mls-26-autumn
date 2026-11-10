"""Put the supplied hw1-asr package (glm_asr_cuda) on sys.path; never modifies it."""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ASR_ROOT = Path(os.environ.get("RAG_ASR_ROOT") or REPO / "hw1-asr").resolve()
if str(ASR_ROOT) not in sys.path:
    sys.path.insert(0, str(ASR_ROOT))
