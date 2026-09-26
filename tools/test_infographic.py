"""Generate one revision infographic locally, outside the web app, to check the
prompt/style changes. Runs the SAME code path as the app (brief summarisation,
image generation, review-and-correct rounds) and saves the PNG next to this file.

Usage (from the StudyApp folder, with your OpenAI key in the environment):

    uv run python tools/test_infographic.py               # built-in sample notes
    uv run python tools/test_infographic.py notes.txt     # your own extracted notes

Cost: one image generation plus up to two review/edit rounds at high quality.
"""
import asyncio
import base64
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SAMPLE_NOTES = """--- Slide 1 ---
Investments: Portfolio Returns and Volatility Risk

--- Slide 2 ---
Individual stock returns. The return on a share combines the dividend received and the change in price.
Log returns make gains and losses symmetric: r = ln((P1 + D1) / P0).
Average percentage returns can mislead when returns are volatile.

--- Slide 3 ---
Volatility. Variance measures how far returns fluctuate around their average.
sigma^2 = sum (r_i - rbar)^2 / (n - 1). Standard deviation is the square root of variance.
Greater standard deviation means greater risk for the investor.

--- Slide 4 ---
Portfolio returns. The portfolio return is the weighted average of the asset returns: r_p = sum w_i r_i.
Portfolio risk depends on the correlation between assets, not just their individual volatilities.
Two-asset portfolio variance: sigma_p^2 = w_A^2 sigma_A^2 + w_B^2 sigma_B^2 + 2 w_A w_B sigma_A sigma_B rho_AB.
Negative correlation reduces portfolio volatility - the benefit of diversification.

--- Slide 5 ---
Sharpe ratio. Compares the excess return of a portfolio with the risk taken: Sharpe = (r_p - r_f) / sigma_p.
A higher Sharpe ratio means a better risk-return trade-off. The efficient frontier shows the portfolios with the highest return for each level of risk.

--- Slide 6 ---
Value at Risk (VaR). VaR estimates the loss that will only be exceeded in the worst 1% of cases.
Volatility scales with the square root of time: sigma_10 = sigma_1 * sqrt(10).
Ten-day 99% VaR = Portfolio value x 10-day volatility x 2.33.
Put options can insure against large market declines; more equity and less borrowing reduce bankruptcy risk.
"""


def load_dotenv_if_present():
    """Minimal .env loader (KEY=VALUE lines) so the test works without exporting the key."""
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip(); value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


async def main(notes_text):
    load_dotenv_if_present()
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set - export it in this shell first.")
        return 1
    from tutor_ai import TutorAI
    tutor = TutorAI()
    tutor.set_context(notes_text, doc_type="lecture_notes")
    started = time.time()
    print("Generating infographic (brief -> image -> review/correct)...")
    image_b64 = await tutor.generate_infographic_async()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_infographic_output.png")
    with open(out_path, "wb") as f:
        f.write(base64.b64decode(image_b64))
    print(f"Saved {out_path} ({len(image_b64) // 1024} KB base64) in {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    text = SAMPLE_NOTES
    if len(sys.argv) > 1:
        with open(sys.argv[1], encoding="utf-8", errors="ignore") as fh:
            text = fh.read()
    sys.exit(asyncio.run(main(text)))
