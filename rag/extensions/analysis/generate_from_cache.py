#!/usr/bin/env python3
"""Answer candidates for selected queries from a variant's transcript cache, judged against
V1's transcript as the question (spec §4.6). No ASR runs here.

Retrieval and generation use the *variant* transcript (what that system would answer);
the row's `transcript` field, which rag/scripts/evaluate_answers.py judges as the question,
is the *V1* transcript, so every variant is judged against the same question.

    python rag/extensions/analysis/generate_from_cache.py --split dev \
        --question-cache rag/.cache/dev_v1.jsonl --transcript-cache rag/.cache/dev_v3_m0.jsonl \
        --ids results/judge/dev_v3_m0_ids.txt --output rag/results/judge-v3_m0-dev.jsonl --device cpu
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RAG = Path(__file__).resolve().parents[2]
if str(RAG) not in sys.path:
    sys.path.insert(0, str(RAG))

from extensions.fast_asr.cachefile import read_cache  # noqa: E402


def build_candidate(qid, split, question, asr_transcript, answer, record_ids, timing) -> dict:
    return {"query_id": qid, "split": split, "transcript": question, "asr_transcript": asr_transcript,
            "answer": answer, "retrieved_record_ids": list(record_ids), "timing": timing}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["practice", "dev"], required=True)
    ap.add_argument("--question-cache", type=Path, required=True, help="V1 transcripts (judged question)")
    ap.add_argument("--transcript-cache", type=Path, required=True, help="variant transcripts (system input)")
    ap.add_argument("--ids", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    if args.output.exists():
        ap.error("Output exists; choose a new filename to preserve prior results")
    ids = [l.strip() for l in args.ids.read_text().splitlines() if l.strip()]
    questions, transcripts = read_cache(args.question_cache), read_cache(args.transcript_cache)
    missing = [q for q in ids if q not in questions or q not in transcripts]
    if missing:
        ap.error(f"{len(missing)} ids missing from a cache, e.g. {missing[:3]}")

    from src.context_builder import SimpleContextBuilder
    from src.data import CourseData
    from src.encoder import LocalEncoder
    from src.generator import ELMGenerator
    from src.retriever import ExactDenseRetriever

    data_dir = RAG / "data" / "rag_corpus"
    data = CourseData(data_dir)
    encoder = LocalEncoder(RAG / "models" / "enc_a", device=args.device)
    retriever = ExactDenseRetriever(data_dir / "embeddings" / "corpus_enc_a.npy",
                                    data_dir / "embeddings" / "corpus_enc_a_ids.json")
    builder, generator = SimpleContextBuilder(), ELMGenerator()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"{len(ids)} ELM generation requests", flush=True)
    with args.output.open("x", encoding="utf-8") as f:
        for qid in ids:
            text = transcripts[qid]
            ret = retriever.retrieve(encoder.encode(text), k=args.k)
            built = builder.build(text, ret["record_ids"], data)
            gen = generator.generate(built["prompt"])
            row = build_candidate(qid, args.split, questions[qid], text, gen["answer"], ret["record_ids"],
                                  {"retrieval_ms": ret["latency_ms"], "generation_ms": gen["latency_ms"]})
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(qid, flush=True)


if __name__ == "__main__":
    main()
