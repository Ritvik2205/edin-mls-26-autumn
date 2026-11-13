# Coursework clarifications (course forum, Nov 2026)

Staff answers to forum questions, recorded so the plan and report follow the current rules.
Where an answer changed the original instructions, the **Rule now** line says what applies.
Thread numbers refer to the course forum. Student names are left out.

Upstream check (2026-11-13): `main` equals `ed-aisys/edin-mls-26-autumn` `main` at `6bbdfbc`
("Update final report instructions in README", 2026-11-06). That commit contains the revised
`hw1-asr/README.md` "What to hand in" section mentioned in #18 and #20.

## Summary: what changed

| Topic | Old instruction | Rule now | Source |
|---|---|---|---|
| HW1 guide questions | Answer the questions at the end of every guide (16) | Answer **at least one question per guide (4 total)**. **Anywhere in the report** (main body or appendix), as long as all four are there and easy to find: clear formatting or a pointer sentence in the introduction. This supersedes #20's 'main body'. | #20, course-team announcement, README |
| "Try this" experiments | At least one per task | **Not required.** They are suggested directions. The one required exploration is "Your investigation": one systems question, ASR *or* RAG. | #20, #25 |
| `make test` / `./run.sh --profile` output | Hand in with the report | Most important results in the main body, the rest in the appendix. | #18, #24, README |
| Kernel changes | README: change anything. Slides: keep names and signatures. | **Keep every given function's name and inputs.** Anything else may change, including kernel bodies such as `linear_tiled_kernel`. | #33 |
| Supplementary code | Unclear what goes beyond the five files | **One single zip** with anything that helps demonstrate the system. It includes the five CUDA files. | #32, README |
| Where RAG experiments run | Unstated | RAG retrieval and investigation experiments **may run on a local CPU** with cached ASR transcripts. Report the hardware, settings and timing boundaries clearly. Use the same hardware for every compared strategy. | #42 |
| `cuda-tutorial/` | Lecture said "hw1 not part of coursework" | That remark meant `cuda-tutorial/`, which is **not** reported. The HW1 ASR implementation **is** reported. | #20 (citing #9) |
| Video structure | Unstated | **No fixed structure or ASR/RAG time split.** It must show an actual run of the system and explain the design choices. Opening with an overview is fine, and so is going task by task. | #21, #22 |
| Video guide location | — | On Learn: Assessment → Group report & individual review folder. | #21 |

## Course-team announcements

### Task requirements and submission materials
- Answer one question from each task. "Try this" is optional.
- The four answers **need not be in the main body**. All four must be in the report and easy to
  find, e.g. through clear formatting or a statement in the introduction.
- Do not resubmit unchanged framework code. The zip holds only files needed to demonstrate the
  system.
- Lengthy `make test` / `--profile` output: put the most relevant parts in the main body and the
  rest in the appendix.

### Video: final two-minute questions (about 1 min each, within the 10-minute total)
1. **Main design choice:** one important decision made when building or experimenting, and why.
2. **Checking your work:** how you checked the system, or your part of it, worked as intended.
   An example is preferred.
- Either ASR or RAG may be discussed; there is no need to cover both. Keep it specific to your
  own work.
- One video per group, and not every member needs to speak. The anonymity rules apply.

## Threads

### #9 / #20: is HW1 ASR part of the coursework?
- The lecture remark ("not part of the coursework") referred to `cuda-tutorial/`. It does not
  go in the report.
- The ASR implementation information must be in the report, as the requirement file says.
- The original instructions required all 16 questions plus Try-this experiments, which would not
  fit in 6 pages. Staff changed this: **answer four questions, one from each guide**. You may
  answer more if it helps the report.
- "Try this" refers to the overall extra-exploration point. That is a single item, ASR or RAG.

### #25: is "Try this" per task still needed?
- Confirmed: only **one systems question** is needed, in ASR or RAG ("Your investigation").
- The per-task Try-this items are now optional exploration directions. Most are too small to stand
  alone, but one can be grown into a full systems question.

### #18: where do the HW1 answers go in the report?
- Put the parts that matter most for the whole system in the main body. The rest goes in the
  appendix. The repository README was updated to say this.

### #24: how to submit `make test` and `./run.sh --profile` output
- Main body: the most important results. Appendix: the auxiliary ones. No separate files.

### #32: what goes in the code zip?
- One single zip as supplementary material. It holds the five CUDA files plus anything that helps
  demonstrate the system, such as RAG extensions and experiment scripts.

### #33: can kernel bodies be rewritten?
- "Please do not change the input and the name of the functions. You are free to do anything else."
- Consequence: rewriting a body (e.g. `linear_tiled_kernel`) is fine. Renaming a function or changing
  its parameters is not.

### #42: running RAG experiments locally on CPU
- Allowed. Keep the cluster for kernel tests, GPU profiling and ASR transcription. Run retrieval and
  investigation experiments locally from cached transcripts.
- The report must state clearly the hardware, settings and timing boundaries, with the same hardware
  for every strategy being compared.

### #21 / #22: video
- No recommended split between ASR and RAG. It depends on the design focus.
- The video must show an actual run of the system and explain the design choices.
- The order is free. A report-style overview first, or a task-by-task walkthrough, are both fine.

## Effect on this project (Option A: ASR redundancy, solo)

- **Report.** The four answers (T1 Q1, T2 Q3, T3 Q2, T4 Q1) go in Appendix A, with a pointer in
  the introduction (spec D5). The main body keeps the key `make test` / `--profile` lines and the
  GPU name. Full logs go in Appendix B.
- **Try-this sweeps (plan Task 5 Step 3)** are now optional. Keep them only where they feed the
  investigation:
  - T3 `TILE` at M = 1 (decode GEMV). It is cited in the spec §4.7 analysis.
  - T4 `ATTN_WARPS` at seq_q = 1.
  Drop T1/T2 unless time allows.
- **`launch_rope_at` (spec §7).** Adding a *new* function in `task1_rope.cu` without touching
  existing names or signatures fits #33 ("free to do anything else"). The padded-scratch fallback
  stays the default until the kernel exists. This is not an explicit approval, so say in the report
  which RoPE path was measured.
- **Zip.** `scripts/make_submission_zip.sh` already builds one zip with the five files, plus
  `rag/extensions`, `supplementary/` and results. That matches #32.
- **Local CPU.** All variants here are ASR variants, so they stay on the pinned GPU. If any
  retrieval/judge reruns happen locally, record the machine and timing boundary in the report, per #42.
