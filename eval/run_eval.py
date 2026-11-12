"""LangSmith evaluation of the YouTube QA bot (the HighlightFinder used by the app).

2026-11-05 · DECISION after Claude's code review: measure the bot with LangSmith.
  Before: quality was checked by ~5 hand-written questions saved as JSON files (no scores, no history).
  Why:    the Project III brief grades "accuracy of the bot in answering questions about YouTube videos"
          and "latency", and asks to "make use of LangSmith for testing and evaluation".
  Gain:   15 questions with reference answers from the real captions, 5 judges, one LangSmith
          experiment per run -> numbers per version, so every later improvement can be proven.

    python eval/run_eval.py                  # LangSmith experiment + Markdown report
    python eval/run_eval.py --local          # report only, no LangSmith
    python eval/run_eval.py --prefix v2-rerank   # name the experiment after the change you made

Judges
    correctness      answer vs reference answer (from the captions)     wrong / missing facts
    relevance        answer vs question                                  off-topic answers
    groundedness     answer vs caption passages the bot retrieved        hallucination
    retrieval        retrieved passages vs question                      search failures
    video_moment     returned clip / source time vs reference window     wrong moment in the video  (YouTube-specific)
"""

import argparse
import json
import os
import socket
import sys
import time
from datetime import datetime
from pathlib import Path

_getaddrinfo = socket.getaddrinfo                     # this PC: IPv4 first, ~1 s instead of ~10 s per API call
socket.getaddrinfo = lambda *a, **k: sorted(_getaddrinfo(*a, **k), key=lambda x: x[0] != socket.AF_INET)

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "eval"
sys.path.insert(0, str(ROOT))

if sys.platform == "win32":                            # keys saved with setx after this shell started
    import winreg
    for _name in ("OPENAI_API_KEY", "LANGSMITH_API_KEY", "LANGSMITH_ENDPOINT"):
        if not os.environ.get(_name):
            try:
                with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as _key:
                    os.environ[_name] = winreg.QueryValueEx(_key, _name)[0]
            except OSError:
                pass

from openai import OpenAI  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

import video_qa  # noqa: E402
from youtube_highlights import HighlightFinder  # noqa: E402

DATASET_NAME = "youtube-qa-testset"
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-4.1-mini")
KEYS = ["correctness", "relevance", "groundedness", "retrieval", "video_moment"]


# ---------- the bot under test ----------

_passages = []
_original_search = video_qa.VideoQA.search


def _capturing_search(self, *args, **kwargs):
    """Same search, but remembers the caption passages the bot saw (the judges need them)."""
    hits = _original_search(self, *args, **kwargs)
    _passages.extend(f"[{h['metadata'].get('trainer', '')} | {h['metadata']['title']} | "
                     f"{video_qa.stamp(h['metadata']['start_seconds'])}] {h['text']}" for h in hits)
    return hits


video_qa.VideoQA.search = _capturing_search
_finder = None


def run_bot(inputs):
    global _finder
    _finder = _finder or HighlightFinder()
    _passages.clear()
    t0 = time.time()
    try:
        result = _finder.find(inputs["question"], trainer=inputs.get("trainer", "Ansh Lamba"))
    except ValueError as exc:                    # the bot's own validation refused to answer: count it, don't crash
        result = {"status": "error", "answer": f"(bot error: {exc})"}
    clips = [result] if result.get("status") == "supported" else result.get("clips", [])
    moments = [{"video_id": c["video_id"], "start": c["start"], "end": c["end"]} for c in clips]
    moments += [{"url": s["url"], "time": s["time"]} for s in result.get("sources", [])]
    docs = list(dict.fromkeys(_passages)) + ([result["evidence"]] if result.get("evidence") else [])
    return {"answer": result.get("answer", ""), "status": result.get("status"), "moments": moments,
            "documents": docs, "seconds": round(time.time() - t0, 1)}


# ---------- the judges ----------

class Grade(BaseModel):
    explanation: str = Field(description="One or two sentences explaining the grade.")
    passed: bool


_judge = None


def judge(instructions, content):
    global _judge
    _judge = _judge or OpenAI()
    return _judge.chat.completions.parse(
        model=JUDGE_MODEL, temperature=0, response_format=Grade,
        messages=[{"role": "system", "content": instructions}, {"role": "user", "content": content}],
    ).choices[0].message.parsed


def correctness(inputs, outputs, reference_outputs):
    g = judge("Grade a student-help bot's answer about a YouTube lecture against a REFERENCE answer written from "
              "the video's captions. Pass if the answer is consistent with the reference and contains its main "
              "points; wording, extra correct detail and app clarifications are fine. Fail on contradictions or if "
              "the main point is missing. If the reference says the topic is not covered, pass only if the bot says "
              "it has no evidence and does not invent an answer. Attribution to the wrong trainer fails.",
              f"QUESTION: {inputs['question']}\n\nREFERENCE: {reference_outputs['answer']}\n\nANSWER: {outputs['answer']}")
    return {"key": "correctness", "score": int(g.passed), "comment": g.explanation}


def relevance(inputs, outputs):
    g = judge("Grade whether the ANSWER addresses the QUESTION. Clearly saying the videos do not cover it is "
              "relevant for a question they cannot answer.",
              f"QUESTION: {inputs['question']}\n\nANSWER: {outputs['answer']}")
    return {"key": "relevance", "score": int(g.passed), "comment": g.explanation}


def groundedness(inputs, outputs):
    g = judge("Grade whether every factual claim in the ANSWER is supported by the caption PASSAGES the bot "
              "retrieved (automatic captions may contain spelling errors). Fail if the answer adds facts that are "
              "not in the passages. A labelled app clarification or 'not covered' statement is allowed.",
              "PASSAGES:\n" + "\n\n".join(outputs["documents"]) + f"\n\nANSWER: {outputs['answer']}")
    return {"key": "groundedness", "score": int(g.passed), "comment": g.explanation}


def retrieval(inputs, outputs, reference_outputs):
    if reference_outputs.get("type") == "unanswerable":
        return {"key": "retrieval", "score": None, "comment": "n/a: nothing relevant exists"}
    g = judge("Grade whether the retrieved caption PASSAGES contain the information needed to answer the QUESTION "
              "(pass even if some passages are unrelated).",
              f"QUESTION: {inputs['question']}\n\nPASSAGES:\n" + "\n\n".join(outputs["documents"]))
    return {"key": "retrieval", "score": int(g.passed), "comment": g.explanation}


def video_moment(inputs, outputs, reference_outputs):
    """YouTube-specific and deterministic: does the bot point the student to the right minute?"""
    window, video = reference_outputs.get("window"), reference_outputs.get("video_id")
    if not window:
        return {"key": "video_moment", "score": None, "comment": "n/a: no single reference moment"}
    lo, hi = window[0] - 30, window[1] + 30                        # 30 s tolerance around the reference passage
    for m in outputs["moments"]:
        if m.get("video_id") == video and m["start"] <= hi and m["end"] >= lo:
            return {"key": "video_moment", "score": 1, "comment": f"clip {m['start']}-{m['end']} s overlaps {window}"}
        if "url" in m and video in m["url"]:
            mm, ss = map(int, m["time"].split(":"))
            if lo <= mm * 60 + ss <= hi:
                return {"key": "video_moment", "score": 1, "comment": f"source at {m['time']} inside {window}"}
    return {"key": "video_moment", "score": 0,
            "comment": f"no clip/source of {video} within {window} s; got {outputs['moments'] or 'none'}"}


EVALUATORS = [correctness, relevance, groundedness, retrieval, video_moment]


def grade(example, outputs):
    out = []
    for evaluator in EVALUATORS:
        names = evaluator.__code__.co_varnames[:evaluator.__code__.co_argcount]
        args = {"inputs": example["inputs"], "outputs": outputs, "reference_outputs": example["outputs"]}
        out.append(evaluator(**{k: v for k, v in args.items() if k in names}))
    return out


# ---------- running and reporting ----------

def load_examples():
    rows = json.loads((EVAL_DIR / "dataset.json").read_text(encoding="utf-8"))
    return [{"inputs": {"question": r["question"], "trainer": r["trainer"]},
             "outputs": {"answer": r["answer"], "type": r["type"], "video_id": r["video_id"], "window": r["window"]}}
            for r in rows]


def row(example, outputs, grades):
    r = {"type": example["outputs"]["type"], "trainer": example["inputs"]["trainer"],
         "question": example["inputs"]["question"], "answer": outputs.get("answer", ""),
         "status": outputs.get("status"), "seconds": outputs.get("seconds", 0)}
    for g in grades:
        r[g["key"]], r[g["key"] + "_why"] = g["score"], g["comment"]
    return r


def run_local(examples):
    rows = []
    for n, ex in enumerate(examples, 1):
        print(f"[{n}/{len(examples)}] {ex['inputs']['question'][:70]}", flush=True)
        out = run_bot(ex["inputs"])
        rows.append(row(ex, out, grade(ex, out)))
    return rows, None


def run_langsmith(examples, prefix):
    from langsmith import Client, evaluate
    client = Client()
    if client.has_dataset(dataset_name=DATASET_NAME):
        ds = client.read_dataset(dataset_name=DATASET_NAME)
        known = {e.inputs["question"] for e in client.list_examples(dataset_id=ds.id)}
    else:
        ds = client.create_dataset(dataset_name=DATASET_NAME,
                                   description="YouTube QA bot: questions with reference answers and video moments")
        known = set()
    new = [e for e in examples if e["inputs"]["question"] not in known]
    if new:
        client.create_examples(dataset_id=ds.id, inputs=[e["inputs"] for e in new],
                               outputs=[e["outputs"] for e in new],
                               metadata=[{"type": e["outputs"]["type"]} for e in new])
    results = evaluate(run_bot, data=DATASET_NAME, evaluators=EVALUATORS,
                       experiment_prefix=f"youtube-qa-{prefix}", max_concurrency=1, metadata={"version": prefix})
    rows = []
    for r in results:
        ex = {"inputs": r["example"].inputs, "outputs": r["example"].outputs}
        grades = [{"key": e.key, "score": e.score, "comment": e.comment} for e in r["evaluation_results"]["results"]]
        rows.append(row(ex, r["run"].outputs or {}, grades))
    return rows, results.experiment_name


def pct(values):
    values = [v for v in values if v is not None]
    return f"{sum(values) / len(values):.0%}" if values else "n/a"


def write_report(rows, prefix, experiment):
    out_dir = EVAL_DIR / "results"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    lines = [f"# YouTube QA bot evaluation, {stamp.replace('_', ' ')}", "",
             f"Version: `{prefix}` · {len(rows)} questions · judge `{JUDGE_MODEL}`"
             + (f" · LangSmith experiment `{experiment}`" if experiment else " · local run"), "",
             "| Judge | Pass rate |", "|---|---|"]
    lines += [f"| {k} | {pct([r.get(k) for r in rows])} |" for k in KEYS]
    secs = [r["seconds"] for r in rows if r.get("seconds")]
    lines += [f"| latency: avg seconds per question | {sum(secs) / max(1, len(secs)):.1f} |", "",
              "## Correctness by question type", "", "| Type | # | Correct |", "|---|---|---|"]
    for t in dict.fromkeys(r["type"] for r in rows):
        g = [r for r in rows if r["type"] == t]
        lines.append(f"| {t} | {len(g)} | {pct([r.get('correctness') for r in g])} |")
    fails = [(r, k) for r in rows for k in KEYS if r.get(k) == 0]
    lines += ["", "## Failed judgements", ""]
    lines += (["| Trainer | Judge | Question | Reason |", "|---|---|---|---|"]
              + [f"| {r['trainer']} | {k} | {r['question']} | {str(r.get(k + '_why', '')).replace('|', '/')} |"
                 for r, k in fails]) if fails else ["None."]
    lines += ["", "## All answers", ""]
    for r in rows:
        marks = " ".join(f"{k}={'n/a' if r.get(k) is None else r.get(k)}" for k in KEYS)
        lines += [f"**[{r['type']} · {r['trainer']}] {r['question']}**  ",
                  f"<sub>status {r['status']} · {r['seconds']} s · {marks}</sub>", "",
                  "> " + str(r["answer"]).replace("\n", "\n> "), ""]
    path = out_dir / f"{stamp}_{prefix}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--local", action="store_true", help="do not send the experiment to LangSmith")
    p.add_argument("--prefix", default="baseline", help="experiment name, e.g. baseline, v2-rerank")
    args = p.parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set.")
    examples = load_examples()
    use_ls = not args.local and bool(os.environ.get("LANGSMITH_API_KEY"))
    print(f"Evaluating {len(examples)} questions ({'LangSmith' if use_ls else 'local'}), version {args.prefix}\n")
    rows, experiment = run_langsmith(examples, args.prefix) if use_ls else run_local(examples)
    report = write_report(rows, args.prefix, experiment)
    print("\nPass rate per judge:")
    for k in KEYS:
        print(f"  {k:<13} {pct([r.get(k) for r in rows])}")
    secs = [r["seconds"] for r in rows if r.get("seconds")]
    print(f"  latency       {sum(secs) / max(1, len(secs)):.1f} s per question")
    print(f"\nReport: {report.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
