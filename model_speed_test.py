
import os
import asyncio
import time

from openai import AsyncOpenAI

GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")

SAMPLE_CONTEXT = """
Introduction to Corporate Finance - Lecture Notes

Capital Structure and the Cost of Capital

The cost of capital is the minimum return that a company must earn on its investments to satisfy
its investors. It is calculated as the weighted average cost of capital (WACC), combining the
cost of equity and the cost of debt.

WACC = (E/V) * Re + (D/V) * Rd * (1 - Tc)

Where:
- E = Market value of equity
- D = Market value of debt
- V = E + D (total firm value)
- Re = Cost of equity
- Rd = Cost of debt
- Tc = Corporate tax rate

The Modigliani-Miller theorem states that, under certain assumptions (no taxes, no bankruptcy
costs, efficient markets), the value of a firm is unaffected by its capital structure.
However, with taxes, debt creates a tax shield that increases firm value.

Dividend Policy
Dividends represent a distribution of profits to shareholders. The dividend irrelevance theory
(Miller and Modigliani, 1961) argues that dividend policy does not affect firm value in perfect
markets. In practice, dividends signal management confidence and attract income-seeking investors.

The Gordon Growth Model estimates share price:
P = D1 / (Re - g)

Where D1 is the next dividend, Re is the required return, and g is the constant growth rate.

Capital Budgeting
NPV (Net Present Value) is the primary tool for investment appraisal. A positive NPV indicates
value creation. IRR (Internal Rate of Return) is the discount rate that makes NPV = 0.
Projects are accepted if IRR exceeds the cost of capital.
"""

PROMPT = (
    "Create a concise executive summary revision sheet of the key concepts, formulas, "
    "and exam-relevant points from these lecture notes. Use clear headings and bullet points."
)

MESSAGES = [
    {
        "role": "system",
        "content": f"You are an expert at creating study aids and revision sheets from academic content.\n\nLecture Notes:\n{SAMPLE_CONTEXT}"
    },
    {
        "role": "user",
        "content": PROMPT
    }
]

MODELS = [
    ("gemini-flash-lite-latest",       "gemini"),
    ("gemini-3.1-flash-lite-preview",  "gemini"),
    ("gpt-5.4-mini",                   "openai"),
    ("gpt-5.4-nano",                   "openai"),
]


async def test_model(name, provider, gemini_client, openai_client):
    client = gemini_client if provider == "gemini" else openai_client
    if not client:
        return name, None, "client not available (missing API key)"

    # gpt-5.4-* don't support system messages — combine into one user message
    if name in ("gpt-5.4-mini", "gpt-5.4-nano"):
        combined = ""
        for m in MESSAGES:
            prefix = "System: " if m["role"] == "system" else ""
            combined += f"{prefix}{m['content']}\n\n"
        messages = [{"role": "user", "content": combined.strip()}]
        extra = {"max_completion_tokens": 4000}
    else:
        messages = MESSAGES
        extra = {"temperature": 0.2, "max_tokens": 4000}

    start = time.perf_counter()
    try:
        response = await asyncio.wait_for(
            client.chat.completions.create(model=name, messages=messages, **extra),
            timeout=90
        )
        elapsed = time.perf_counter() - start
        content = response.choices[0].message.content or ""
        return name, elapsed, f"OK — {len(content)} chars"
    except Exception as e:
        elapsed = time.perf_counter() - start
        return name, elapsed, f"FAILED: {e}"


async def main():
    gemini_client = AsyncOpenAI(
        api_key=GEMINI_KEY,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        max_retries=0,
    ) if GEMINI_KEY else None

    openai_client = AsyncOpenAI(
        api_key=OPENAI_KEY,
        max_retries=0,
    ) if OPENAI_KEY else None

    print("\n=== Model Speed Test ===")
    print(f"Context: {len(SAMPLE_CONTEXT)} chars | Prompt: {len(PROMPT)} chars\n")

    tasks = [test_model(name, provider, gemini_client, openai_client) for name, provider in MODELS]
    results = await asyncio.gather(*tasks)

    print(f"{'Model':<40} {'Time':>8}   Result")
    print("-" * 75)
    for name, elapsed, status in sorted(results, key=lambda r: r[1] if r[1] is not None else 999):
        t = f"{elapsed:.2f}s" if elapsed is not None else "  n/a"
        print(f"{name:<40} {t:>8}   {status}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
