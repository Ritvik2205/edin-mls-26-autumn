# Video script and recording directions

Target: 9:40 total. Sections A–G end by 7:45, then Question 1 and Question 2 fill the final
two minutes. Hard limit 10:00. Narration is written at about 140 words per minute; read it at a
normal pace and do not add to it.

Rules this follows (Submission Requirements, Video Production Guide, forum #21/#22):

- at most 10 minutes; up to 8 for presentation and demonstration; the final 2 for the two
  released questions, each introduced aloud as "Question 1" / "Question 2";
- an actual recorded run of the system with its input, configuration and output, and an
  explanation of what the output shows; any shortened wait is captioned;
- screen recording with narration only: no webcam, no name, UUN, Group ID, email, hostname,
  home path, browser profile, notification or API key on screen or in metadata;
- file named `video.mp4`, H.264 + AAC, text readable at 1080p.

---

## 1. What you need on screen

Three sources only. Nothing else is ever visible.

1. **The report PDF** (`report/asr-redundancy/main.pdf`) in Preview, full screen, zoomed so one
   table or figure fills the width. It already has the anonymous title. **Never scroll to the
   last page** (declaration page with your details). Stop at the appendix.
2. **One terminal** on a cluster GPU node, font 20 pt or larger, about 110 columns, cleaned as
   in section 2.
3. **One code view**: `rag/extensions/fast_asr/tests/test_equivalence.py` open in the same
   terminal with `less`, for Question 2.

## 2. Preparation (do before recording, none of it is filmed)

On the Mac:

- System Settings → Focus → Do Not Disturb on. Quit Mail, Slack, WhatsApp, browsers.
- Hide the menu bar and Dock (System Settings → Control Centre → "Automatically hide and show
  the menu bar": Always), or record a selected portion that excludes them. The menu bar shows
  your account name in some setups.
- Terminal → Settings → Profiles: font size 20, and untick everything under "Window → Title"
  so the title bar shows no user, host or path. Simplest: record a selected region that
  excludes the title bar.

On the cluster (wait until no other GPU job of yours is running: one GPU job at a time):

```bash
ssh <cluster>                       # not filmed: the login banner shows your account
cd ~/edin-mls-26-autumn
srun -p Teaching --gres=gpu:nvidia_geforce_rtx_2080_ti:1 --cpus-per-task=2 --mem=16G --time=01:00:00 --pty bash
export PATH=/opt/cuda-12.8.0/bin:$PATH
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mls-asr
make -C hw1-asr lib
export PS1='$ '                     # prompt without user, host or path
export RAG_ASR_PYTHON=$(which python)
mkdir -p /tmp/demo && rm -f /tmp/demo/*
anon() { sed -u -E 's#/home/[A-Za-z0-9_]+#~#g; s#[a-z0-9]+\.inf\.ed\.ac\.uk#gpu-node#g'; }
clear
```

`anon` removes home paths and the node name from anything a command prints. Every demo command
below is piped through it. Do not run `hostname`, `whoami`, `pwd`, `squeue`, `git log` or
`cat rag/.env` while recording.

**Rehearse every demo command once before recording.** It warms the file cache and tells you
whether anything prints an identifying string. If the full-pipeline command (D4) fails because
the ELM key is not configured on the cluster, drop D4 and use the fallback line given there;
do not fix the key on camera.

## 3. How to record

Use the macOS recorder (no install needed):

1. Press **Cmd+Shift+5** → "Record Selected Portion" → drag the box over the terminal or the
   PDF content only (no title bar, no menu bar).
2. Options → Microphone: your headset or built-in mic. Options → untick "Show Mouse Clicks"
   is fine either way.
3. Record a 10-second test, play it back: voice audible, smallest text readable.
4. Record **each section as its own clip** (A, B, C, D, E, F, G, Q1, Q2). A mistake then costs
   one clip. Stop with the stop button in the menu bar or Cmd+Ctrl+Esc. Clips land on the
   Desktop as `.mov`; move them to `video/clips/` and name them `A.mov` … `Q2.mov`.

Joining, captioning a shortened wait, and stripping metadata are in section 5.

---

## 4. Script

### A. Question and system (0:00–0:50) — screen: report page 1, title and introduction

> This project is called "Removing redundant computation from a speech-to-text front end".
> The system is an audio question-answering pipeline. A spoken question goes into a
> speech recogniser, GLM-ASR-Nano, running on CUDA kernels we wrote by hand. The transcript is
> embedded, the twenty-four thousand passage corpus is searched exactly, and a language model
> writes the answer from the top passages.
>
> Our question is: how much of the speech stage's latency is work whose result is already
> known, and can we remove it without changing what the pipeline answers? We found two such
> redundancies. The decoder re-runs the whole prefix for every new token, and the encoder
> always processes thirty seconds of input for questions that last about five.

### B. Baseline and kernels (0:50–1:40) — screen: Table 1 (variants), then scroll to Appendix A heading

> There are four variants. V0 is the supplied PyTorch reference. V1 is the supplied model
> running on our own kernels from the first homework: rotary position embedding, warp and
> block reductions for RMS norm, a tiled matrix multiply, and attention. V1 is our baseline.
> V2 adds a key-value cache to the decoder. V3 adds encoder trimming on top of V2.
>
> All four kernel tasks pass the supplied tests on an RTX 2080 Ti, and every number in this
> video comes from that one GPU type. The four guide questions are answered in Appendix A.
> We never modify the supplied model code: our variants wrap the loaded model and call the
> same operation layer, so the same code runs on our kernels or on PyTorch.

### C. The two changes (1:40–3:10) — screen: Section 3 design text, then Table 2 (cost model)

> First, the key-value cache. In the baseline, generating token number t runs all twenty-eight
> decoder layers over the whole prefix again, around sixty to a hundred and thirty positions.
> But attention is causal, so the keys and values of earlier positions never change. V2 stores
> them once and then runs a single position per step. That needed one new kernel entry point:
> rotary embedding at a position offset, because the supplied kernel always starts at zero.
> This change is exact. We predicted, and then tested, that the token probabilities are
> bit-identical to the baseline, because each of our kernels computes a row independently of
> how many rows there are.
>
> Second, trimming. The encoder has no padding mask, so it spends most of its time on silence.
> V3 keeps only the real audio frames, rounded up to a multiple of eight, optionally with a
> margin. It never cuts real audio, so the number of audio tokens the decoder sees is
> unchanged. It is approximate, because real frames can no longer attend to the padding.
>
> Before timing anything, we committed a cost model, shown here: a bandwidth floor for one
> decode step, nine point six milliseconds to stream every decoder weight once, and a predicted
> encoder cost ratio for each trimming setting.

### D. Demonstration (3:10–5:10) — screen: terminal

Say the configuration aloud while the first command is on screen.

**D1. Baseline, three dev queries** (type, run; cut the model-load wait):

```bash
python rag/extensions/fast_asr/bench_asr.py --split dev --variant v1 --limit 3 --warmup 1 \
  --out-csv /tmp/demo/v1.csv --cache /tmp/demo/v1.jsonl 2>&1 | anon
```

> This is a real run on a cluster node with an RTX 2080 Ti, all four kernel groups on CUDA.
> The input is the first three audio questions of the dev split, sixteen kilohertz mono
> speech. This command is the baseline, V1. The model loads once, which is excluded from
> timing, one warm-up query is discarded, and each line then reports the transcription time
> for one question: around five seconds each.

**D2. KV cache plus trimming** (same three queries):

```bash
python rag/extensions/fast_asr/bench_asr.py --split dev --variant v3 --margin 0 --limit 3 --warmup 1 \
  --out-csv /tmp/demo/v3.csv --cache /tmp/demo/v3.jsonl 2>&1 | anon
```

> Same questions, same node, now with the cache and with the encoder trimmed to the real
> length. Each is now around one second.

**D3. Compare input, transcript and time side by side:**

```bash
python - <<'EOF'
import csv
for name in ("v1", "v3"):
    for r in csv.DictReader(open(f"/tmp/demo/{name}.csv")):
        if r["pass"] == "0":
            print(f'{name}  {float(r["duration_s"]):4.1f}s audio  enc {float(r["encode_ms"]):6.0f} ms  '
                  f'dec {float(r["decode_ms"]):6.0f} ms  total {float(r["asr_ms"]):6.0f} ms  {r["transcript"]}')
EOF
```

> Here are both runs together. For each question: audio length, encoder time, decoder time,
> total, and the transcript. The encoder drops from about one point seven seconds to a quarter
> of a second, the decoder from over three seconds to under one, and the transcripts are the
> same. This recording is not the benchmark; the measured results come from full runs of all
> one hundred and twenty-five dev questions, which I show next.

**D4. Whole pipeline, one question** (cut the wait; caption it):

```bash
python rag/extensions/run_audio_query_fast.py --split dev --device cpu --variant v3 --margin 0 2>&1 | anon
```

> Finally the whole pipeline on one spoken question with the fast variant: the transcript,
> the retrieved passages, the generated answer, and the per-stage timings. Generation by the
> language model is now the largest stage.

Scroll so `transcript`, `answer` and `timings` are each visible for two seconds.
Fallback if this command cannot reach the language model: skip D4, and after D3 say
"The transcripts then go to the unchanged retrieval and generation stages" (saves 20 s).

### E. Results (5:10–6:40) — screen: Table 3 (dev results), then Figure 1

> The measured results, on the dev split, one hundred and twenty-five questions. The baseline
> takes four point eight seconds per question on average. The cache alone brings that to two
> point five seconds, one point nine times faster, with every transcript identical and the
> probabilities bit-identical. A decode step falls from a hundred and eighty-seven
> milliseconds to thirty-eight.
>
> With decoding cheap, the untouched encoder is sixty-nine percent of the time. Trimming it to
> the real length brings the total to about one second: four point six times faster than the
> baseline. Eleven of the hundred and twenty-five transcripts change. We read all of them.
> Most differ in spacing or punctuation; the rest are the same sounds written differently,
> such as "H one B" becoming "H1B" as one token. Recall at ten against the relevance labels is
> unchanged, zero point four seven five, and the judged answer quality on the changed
> questions is within the judge's own noise. So we report these as changes, not errors: there
> are no reference transcripts.
>
> The figure on the right shows the speed-up per question. The cache helps long questions
> more, because the baseline's recomputation grows with prefix length. Trimming removes the
> fixed encoder cost, so short questions gain too.

### F. Why it stops there (6:40–7:25) — screen: Discussion section, launch-cost paragraph

> After the cache, every decoder matrix multiply has a single row, which our tiled kernel
> handles badly. We wrote two vector-matrix kernels; the better one is within six percent of
> cuBLAS and brings the total to five point three times faster than the baseline. But a step
> is still three times above our bandwidth floor. We measured why: we split each step into the
> time the host spends issuing its three hundred and forty kernel launches, and the time the
> GPU still has work queued afterwards. The host part is twenty-nine milliseconds; the GPU
> wait is under one. Ninety-seven percent of the step is the Python host, not the kernels.
> Our floor model had no host term, and that is where it was wrong.

### G. Limits (7:25–7:45) — screen: Limitations paragraph

> The limits. One model and one GPU type. Each question is timed once per run, with a
> repeated subset to check variability. The cache's gain is specific to our slower matrix
> multiply: on the PyTorch path it does not help, which we measured and report. And the answer
> judge covers eighty questions, so it can only exclude large quality changes.

### Question 1 (7:45–8:40) — screen: Table 1, then Table 3 rows V1 and V2

> **Question 1: the main design choice.**
> The most important decision was to do the exact optimisation first and to make it provably
> exact, before allowing anything approximate. We could have combined the cache, trimming and
> a faster kernel in one step and reported one large speed-up. Instead, V2 was required to
> reproduce the baseline's token probabilities bit for bit, and only then did V3 add the one
> approximate change. The reason is attribution. When eleven transcripts changed under
> trimming, we already knew the cache could not be responsible, so every change belongs to
> one mechanism: real frames no longer attending to padding. The same choice shaped the
> trimming rule itself. It may remove padding but never real audio, so the number of audio
> tokens and the prompt stay the same, and a unit test asserts that for the longest clip.
> The cost of this choice is speed we did not take: the faster of our two vector kernels
> reorders a sum, so we kept a second, bit-identical kernel alongside it and report both.

### Question 2 (8:40–9:40) — screen: `less rag/extensions/fast_asr/tests/test_equivalence.py` at `test_kv_bit_identical_on_student_kernels`, then Table 3 lower block

> **Question 2: checking the work.**
> We checked at three levels. First, unit tests: this one runs the baseline and the cached
> decoder on the same audio and requires equal tokens and bitwise-equal probabilities at
> every step. Another checks that our timing driver reproduces the supplied generate function
> exactly, so the baseline we time is the supplied one. The suite has fifty-two tests and
> runs on the same GPU type as the experiments.
>
> Second, a control. We repeated the variants on the PyTorch path, with none of our kernels.
> The same eleven transcripts change under trimming there too, which shows the changes come
> from the model and not from a bug in our kernels.
>
> Third, a concrete case where checking changed a result. The dev runs had been placed on
> four different nodes. We reran all five on one node. Transcripts were identical, but the
> baseline was three point eight percent faster there, so we recomputed every speed-up from
> the same-node rerun rather than keep the more flattering numbers.

---

## 5. Assemble, check, export

Put the clips in `video/clips/`. For any clip where you waited for a model load, cut the wait
and burn a caption over the first two seconds after the cut. Example for `D.mov` with a wait
from 0:20 to 1:35:

```bash
cd video/clips
ffmpeg -i D.mov -t 20 -c:v libx264 -crf 20 -r 30 -c:a aac D_1.mp4
ffmpeg -ss 95 -i D.mov -vf "drawtext=text='Waiting time shortened':x=40:y=40:fontsize=42:fontcolor=white:box=1:boxcolor=black@0.7:enable='lt(t,3)'" \
  -c:v libx264 -crf 20 -r 30 -c:a aac D_2.mp4
```

Convert the others the same way without the filter (`ffmpeg -i A.mov -c:v libx264 -crf 20 -r 30 -c:a aac A.mp4`),
scale all clips to one size if they differ (add `-vf scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2`),
then join and strip metadata:

```bash
printf "file '%s'\n" A.mp4 B.mp4 C.mp4 D_1.mp4 D_2.mp4 E.mp4 F.mp4 G.mp4 Q1.mp4 Q2.mp4 > list.txt
ffmpeg -f concat -safe 0 -i list.txt -c:v libx264 -crf 22 -r 30 -c:a aac -map_metadata -1 -movflags +faststart ../video.mp4
ffprobe -v error -show_entries format=duration:format_tags -of default=nw=1 ../video.mp4
```

Final checks before upload:

- [ ] duration ≤ 600 s, and Question 1 + Question 2 are the last two minutes;
- [ ] `ffprobe` shows no title/author/artist tags; file is `video.mp4`;
- [ ] watch it once at full size: no name, UUN, hostname, `/home/...` path, notification, menu
      bar, last page of the report, or `.env` content anywhere;
- [ ] narration audible throughout; terminal and tables readable;
- [ ] every cut wait has the "Waiting time shortened" caption;
- [ ] upload through the Learn route, then play the uploaded copy.
