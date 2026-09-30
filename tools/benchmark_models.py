"""Speed test of candidate primary models through the app's own code paths.

Runs, for each model, the executive summary (non-streaming), a chat question
(streaming: time to first token and total) and multiple-choice quiz generation,
interleaving the models so a slow spell on the API does not bias one side.
Retries are disabled so a slow call is measured, not hidden.

Usage (from StudyApp, key in .env or the environment):
    uv run python tools/benchmark_models.py                  # gpt-6-sol vs gpt-6.1-sol, 2 rounds
    uv run python tools/benchmark_models.py gpt-6-sol gpt-6-luna --rounds 3
    uv run python tools/benchmark_models.py --effort low    # also compare reasoning effort
"""
import argparse
import asyncio
import logging
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from test_infographic import SAMPLE_NOTES, load_dotenv_if_present  # noqa: E402

CHAT_QUESTION = "Explain the Sharpe ratio in two or three sentences and say why a higher value is better."


async def time_summary(tutor):
    t0 = time.perf_counter()
    out = await tutor.generate_cheat_sheet_async()
    return time.perf_counter() - t0, None, len(out or "")


async def time_chat(tutor):
    t0 = time.perf_counter()
    first = None
    chars = 0
    async for chunk in tutor.get_response_stream_async(CHAT_QUESTION):
        if first is None:
            first = time.perf_counter() - t0
        chars += len(chunk)
    return time.perf_counter() - t0, first, chars


async def time_quiz(tutor):
    t0 = time.perf_counter()
    out = await tutor.generate_retrieval_quiz_async()
    return time.perf_counter() - t0, None, len(out or [])


TASKS = [("executive summary", time_summary), ("chat (stream)", time_chat), ("quiz (15 MCQ)", time_quiz)]


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*", default=["gpt-6-sol", "gpt-6.1-sol"])
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--effort", default=None, help="reasoning effort to use for all calls (default: app setting)")
    args = ap.parse_args()

    load_dotenv_if_present()
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set"); return 1

    import tutor_ai
    from tutor_ai import TutorAI
    tutor_ai.PRIMARY_MAX_ATTEMPTS = 1          # measure, do not retry
    tutor_ai.NO_SYSTEM_MESSAGE_MODELS = tuple(set(tutor_ai.NO_SYSTEM_MESSAGE_MODELS) | set(args.models))
    if args.effort:
        tutor_ai.DEFAULT_REASONING_EFFORT = args.effort
    print(f"Models: {args.models} | rounds: {args.rounds} | reasoning effort: {tutor_ai.DEFAULT_REASONING_EFFORT}\n")

    results = {m: {t: [] for t, _ in TASKS} for m in args.models}
    firsts = {m: [] for m in args.models}
    for rnd in range(1, args.rounds + 1):
        for model in args.models:
            tutor_ai.MODEL_PRIMARY = model
            tutor_ai.MODEL_FALLBACK = model    # never silently measure the other model
            tutor = TutorAI(); tutor.set_context(SAMPLE_NOTES, doc_type="lecture_notes")
            for name, fn in TASKS:
                try:
                    total, first, size = await fn(tutor)
                    results[model][name].append(total)
                    if first is not None:
                        firsts[model].append(first)
                    extra = f", first token {first:5.1f}s" if first is not None else ""
                    print(f"round {rnd} | {model:12s} | {name:18s} | {total:6.1f}s{extra} | size {size}")
                except Exception as e:
                    print(f"round {rnd} | {model:12s} | {name:18s} | FAILED: {type(e).__name__}: {str(e)[:120]}")

    print("\nSUMMARY (mean seconds over successful runs)")
    print(f"{'task':20s}" + "".join(f"{m:>16s}" for m in args.models))
    for name, _ in TASKS:
        row = f"{name:20s}"
        for m in args.models:
            xs = results[m][name]
            row += f"{statistics.mean(xs):14.1f}s " if xs else f"{'n/a':>16s}"
        print(row)
    row = f"{'chat first token':20s}"
    for m in args.models:
        xs = firsts[m]
        row += f"{statistics.mean(xs):14.1f}s " if xs else f"{'n/a':>16s}"
    print(row)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    sys.exit(asyncio.run(main()))
