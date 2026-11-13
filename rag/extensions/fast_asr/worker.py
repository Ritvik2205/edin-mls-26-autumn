"""Variant ASR worker: the rag/src/asr_worker.py JSON-lines protocol, backed by FastGlmAsr.

{"ready": true} once loaded; then {"audio_path": ...} -> {"transcript": ..., "timing": {...}}
or {"error": ...}. All other output goes to stderr so stdout stays protocol-only.
"""
import argparse
import json
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # rag/ -> `extensions.*`


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--asr-root", type=Path, required=True)
    ap.add_argument("--cuda", default="all")
    ap.add_argument("--max-new-tokens", type=int, default=100)
    ap.add_argument("--variant", choices=["v1", "v2", "v3"], default="v2")
    ap.add_argument("--margin", type=float)
    ap.add_argument("--bucket", type=float)
    args = ap.parse_args()
    os.environ["RAG_ASR_ROOT"] = str(args.asr_root.resolve())
    try:
        with redirect_stdout(sys.stderr):
            import torch
            from extensions.fast_asr import _paths  # noqa: F401
            from glm_asr_cuda import ops
            from extensions.fast_asr.bench_asr import GpuTranscriber
            from extensions.fast_asr.kv_model import load_fast
            if not torch.cuda.is_available():
                raise RuntimeError("Run ASR on a CUDA GPU node")
            ops.set_cuda_groups(args.cuda)
            if ops.USE_CUDA and not ops.library_path().is_file():
                raise RuntimeError("Run make lib in hw1-asr on this GPU node first")
            fast, processor = load_fast()
            tr = GpuTranscriber(fast, processor, args.variant, args.margin, args.bucket, args.max_new_tokens)
        print(json.dumps({"ready": True}), flush=True)
    except Exception as exc:
        print(json.dumps({"error": str(exc)}), flush=True)
        return 1
    for line in sys.stdin:
        try:
            request = json.loads(line)
            with redirect_stdout(sys.stderr):
                m = tr.transcribe({"path": request["audio_path"]})
            timing = {k: m[k] for k in ("feat_ms", "encode_ms", "prefill_ms", "decode_ms", "n_tokens", "enc_frames")}
            response = {"transcript": m["transcript"], "timing": timing}
        except Exception as exc:
            response = {"error": str(exc)}
        print(json.dumps(response, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
