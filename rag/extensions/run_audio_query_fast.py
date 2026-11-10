"""rag/scripts/run_audio_query.py with a selectable ASR variant (v1 recompute, v2 KV cache,
v3 KV cache + encoder trimming). Run from the repository root on a GPU node."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]  # rag/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json

from extensions.fast_asr.adapter import FastASRAdapter
from src.context_builder import SimpleContextBuilder
from src.data import CourseData
from src.encoder import LocalEncoder
from src.generator import ELMGenerator
from src.rag import RAGPipeline
from src.retriever import ExactDenseRetriever


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    selection = ap.add_mutually_exclusive_group()
    selection.add_argument("--audio", help="Path to a 16 kHz mono PCM16 WAV")
    selection.add_argument("--query-id", help="Select a query from the course manifest")
    ap.add_argument("--split", choices=["practice", "dev"], default="practice")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--device", default=None)
    ap.add_argument("--variant", choices=["v1", "v2", "v3"], default="v2")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--margin", type=float)
    g.add_argument("--bucket", type=float)
    args = ap.parse_args()
    if args.k < 1:
        ap.error("--k must be positive")

    data_dir = ROOT / "data" / "rag_corpus"
    data = CourseData(data_dir)
    query_id = None
    if args.audio:
        audio = Path(args.audio)
    else:
        items = data.audio_items(args.split)
        if args.query_id:
            items = [item for item in items if item["query_id"] == args.query_id]
        if not items:
            ap.error("No matching query in the selected split")
        query_id = items[0]["query_id"]
        audio = data_dir / items[0]["audio_file"]
    try:
        asr = FastASRAdapter(variant=args.variant, margin_s=args.margin, bucket_s=args.bucket)
    except ValueError as e:
        ap.error(str(e))
    asr.prepare()
    pipeline = RAGPipeline(
        asr=asr,
        encoder=LocalEncoder(ROOT / "models" / "enc_a", device=args.device),
        retriever=ExactDenseRetriever(data_dir / "embeddings" / "corpus_enc_a.npy",
                                      data_dir / "embeddings" / "corpus_enc_a_ids.json"),
        data=data,
        context_builder=SimpleContextBuilder(),
        generator=ELMGenerator(),
    )
    result = pipeline.answer_audio(audio, k=args.k)
    result.update(query_id=query_id, audio_file=str(audio), asr_variant=args.variant,
                  asr_setting={"margin_s": args.margin, "bucket_s": args.bucket})
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
