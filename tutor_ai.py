
import os
import asyncio
import json
import logging
import random
import re

import openai
from openai import AsyncOpenAI

# Maximum number of document characters sent to the model as context.
# Whole-document features (summary, key concepts, quiz, equation extraction)
# get a large budget - their outputs are cached per document in app.py, so the
# cost is paid roughly once per upload. This covers all but extreme decks.
MAX_CONTEXT_CHARS = 320000

# Chat resends context on every message, so it uses a smaller budget; when a
# document exceeds it, the pages most relevant to the student's question are
# selected (see _select_relevant_context) so any page remains reachable and
# citable regardless of document length.
CHAT_CONTEXT_CHARS = 120000

# Answer checking grades a self-contained challenge question; the notes are
# only needed for terminology, so a small slice suffices.
ANSWER_CHECK_CONTEXT_CHARS = 20000

# Page/slide markers emitted by pdf_processor ("--- Page 4 ---", "--- Slide 12 ---").
_PAGE_MARKER_SPLIT = re.compile(r'(?m)^(?=--- (?:Page|Slide) \d+ ---$)')

# Words too common to signal relevance when matching a question against pages.
_STOPWORDS = frozenset((
    'the', 'and', 'for', 'with', 'this', 'that', 'what', 'how', 'why', 'when',
    'where', 'which', 'does', 'can', 'could', 'would', 'should', 'you', 'your',
    'are', 'was', 'were', 'has', 'have', 'had', 'not', 'but', 'they', 'them',
    'there', 'their', 'then', 'than', 'its', 'about', 'from', 'into', 'out',
    'please', 'explain', 'tell', 'show', 'give', 'help', 'notes', 'lecture',
    'mean', 'means', 'more', 'some', 'any', 'all', 'also', 'just', 'like',
))

# Primary and fallback model names used across all features.
MODEL_PRIMARY = "gpt-5.6-terra"
MODEL_FALLBACK = "gpt-5.6-luna"

# Models in this family do not support system messages (all messages must be
# combined into a single user message), use max_completion_tokens instead of
# max_tokens, and reject non-default temperature values.
NO_SYSTEM_MESSAGE_MODELS = ("gpt-5.6-terra", "gpt-5", "gpt-5-mini", "gpt-5.6-luna", "gpt-5.4")

# Lazily-initialized module-level AsyncOpenAI client. TutorAI is constructed
# per request in app.py, so the client is shared across instances to avoid
# rebuilding HTTP connection pools on every request.
_async_openai_client = None


def _get_async_openai_client():
    """Return the shared AsyncOpenAI client, creating it on first use."""
    global _async_openai_client
    if _async_openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
        # max_retries=2: the SDK performs its own backoff for transient
        # connection/rate-limit/5xx errors.
        _async_openai_client = AsyncOpenAI(api_key=api_key, max_retries=2)
    return _async_openai_client


# --- Deterministic formatting normalizer for summary / key concepts output ---
# The models usually follow the prompt's markdown structure but occasionally
# emit it as plain text (headings without ***, categories without bold,
# "One:" instead of "1."). Prompt rules reduce this but cannot eliminate it,
# so the known structure is repaired server-side on the way out.

_STUDY_HEADINGS = frozenset((
    'OVERVIEW', 'KEY CONCEPTS', 'KEY CONCEPTS EXPLAINED',
    'EXECUTIVE SUMMARY', 'SUMMARY', 'ESSAY QUESTION',
))

# Essay section headings whose canonical style is **bold** (not ***bold-italic***).
_ESSAY_BOLD_HEADINGS = frozenset((
    'MAIN QUESTION', 'SUB-QUESTIONS', 'SUB QUESTIONS', 'ASSESSMENT CRITERIA',
))

_NUMBER_WORDS = {
    'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
    'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10',
}

_CATEGORY_LINE_RE = re.compile(r'^(\d+)[.):]\s+(.+?)\s*$')
_WORD_CATEGORY_RE = re.compile(
    r'^(One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten)[.:]\s+(.+?)\s*$', re.IGNORECASE)
_CONCEPT_BULLET_RE = re.compile(r'^(\s*)[-•]\s+(?![*\d])([^:*`]{2,60}):\s+(.*)$')


def _normalize_study_line(line, mode='study'):
    """Repair one line of summary/key-concepts/essay output to the house markdown style.

    mode='study': numbered headers repair to **1. Name** (summary, key concepts).
    mode='essay': numbered sub-questions repair to *1: Question* (essay canonical).
    """
    s = line.strip()
    if not s:
        return line

    # Heading lines: OVERVIEW / ESSAY QUESTION etc. -> ***HEADING***;
    # essay section headings (MAIN QUESTION, SUB-QUESTIONS, ...) -> **HEADING**
    bare = s.strip('#*').strip()
    key = bare.rstrip(':').upper()
    if key in _STUDY_HEADINGS:
        return f'***{bare}***'
    if key in _ESSAY_BOLD_HEADINGS or key.startswith('ASSESSMENT CRITERIA'):
        return f'**{bare}**'

    # Numbered headers/sub-questions only ever start at column 0; indented
    # lines are guidance/sub-bullets and must not be touched.
    indented = line[:1].isspace()

    # Headers numbered as words: "One: ..." -> digits with house emphasis.
    m = _WORD_CATEGORY_RE.match(s)
    if m and '*' not in s and not indented:
        num = _NUMBER_WORDS[m.group(1).lower()]
        if mode == 'essay':
            return f'*{num}: {m.group(2)}*'
        if len(s) < 90:
            return f'**{num}. {m.group(2)}**'

    # Digit-numbered headers missing emphasis.
    m = _CATEGORY_LINE_RE.match(s)
    if m and '*' not in s and not indented:
        if mode == 'essay':
            return f'*{m.group(1)}: {m.group(2)}*'
        if len(s) < 90:
            return f'**{m.group(1)}. {m.group(2)}**'

    # Concept bullets missing the italic label: "- Concept: text" -> "- *Concept:* text"
    m = _CONCEPT_BULLET_RE.match(line)
    if m:
        return f'{m.group(1)}- *{m.group(2).strip()}:* {m.group(3)}'

    return line


def _is_overview_heading(line):
    return line.strip().strip('#*').strip().rstrip(':').upper() == 'OVERVIEW'


def _is_prose_line(line):
    """A plain sentence line: not blank and not a heading/bullet/numbered header."""
    s = line.strip()
    if not s or s[0] in '*#-•':
        return False
    if _CATEGORY_LINE_RE.match(s) or _WORD_CATEGORY_RE.match(s):
        return False
    return True


class _StudyFormattingNormalizer:
    """Stateful line-by-line formatter shared by the streaming and
    non-streaming paths. Applies _normalize_study_line to each line and joins
    the OVERVIEW section back into a single paragraph when the model emits it
    one sentence per line (the frontend renders each newline literally).

    feed() returns the text to append for one input line, including any
    separator owed from earlier lines; flush() returns whatever is still
    pending at end of input. Working line-by-line keeps streaming incremental.
    """

    def __init__(self, mode='study'):
        self.mode = mode
        self._started = False        # any line emitted yet
        self._pending_blanks = 0     # blank lines seen but not yet emitted
        self._in_overview = False
        self._prev_prose = False     # last non-blank line was overview prose

    def feed(self, raw_line):
        line = _normalize_study_line(raw_line, self.mode)
        s = line.strip()

        if not s:
            # Defer blank lines: inside the overview they are dropped if
            # prose continues (sentence-per-paragraph output); otherwise they
            # are re-emitted before the next line or at flush().
            self._pending_blanks += 1
            return ''

        if self._in_overview and self._prev_prose and _is_prose_line(line):
            # Continuation sentence: join into the same paragraph.
            self._pending_blanks = 0
            return ' ' + s

        sep = '\n' * (self._pending_blanks + (1 if self._started else 0))
        self._pending_blanks = 0
        self._started = True

        if _is_overview_heading(line):
            self._in_overview = True
            self._prev_prose = False
        elif self._in_overview:
            if _is_prose_line(line):
                self._prev_prose = True
            else:
                # Any structured line (next heading, category, bullet) ends
                # the overview section.
                self._in_overview = False
                self._prev_prose = False

        return sep + line

    def flush(self):
        out = '\n' * self._pending_blanks
        self._pending_blanks = 0
        return out


def _normalize_study_formatting(text, mode='study'):
    """Apply the stateful line normalizer across a complete response."""
    if not text:
        return text
    norm = _StudyFormattingNormalizer(mode)
    parts = [norm.feed(line) for line in text.split('\n')]
    parts.append(norm.flush())
    return ''.join(parts)


async def _normalize_study_stream(agen, mode='study'):
    """Line-buffered streaming wrapper around _StudyFormattingNormalizer."""
    norm = _StudyFormattingNormalizer(mode)
    buffer = ''
    ends_with_newline = False
    async for chunk in agen:
        buffer += chunk
        while '\n' in buffer:
            line, buffer = buffer.split('\n', 1)
            ends_with_newline = True
            out = norm.feed(line)
            if out:
                yield out
        if buffer:
            ends_with_newline = False
    if buffer:
        out = norm.feed(buffer)
        if out:
            yield out
    elif ends_with_newline:
        norm.feed('')  # re-create the final newline consumed by the last split
    tail = norm.flush()
    if tail:
        yield tail


def _strip_code_fences(text):
    """Remove a wrapping markdown code fence (e.g. ```markdown ... ```) that models occasionally add.

    Handles language tags containing digits (e.g. ```json5), trailing spaces
    after the tag, \\r\\n line endings, and short stray text after the closing
    fence.
    """
    if not text:
        return text
    text = text.strip()
    opening = re.match(r'^```[A-Za-z0-9_+\-]*[ \t]*\r?\n', text)
    if opening:
        text = text[opening.end():]
        # Remove the closing fence line plus any short trailing remark after it.
        closing = re.search(r'\r?\n```[ \t]*(\r?\n.{0,200})?\s*$', text, flags=re.S)
        if closing:
            text = text[:closing.start()]
    else:
        # No opening fence; still remove a dangling closing fence at the end.
        text = re.sub(r'\r?\n?```[ \t]*$', '', text)
    return text.strip()


class TutorAI:

    def __init__(self):

        # Fail fast if the API key is missing. The AsyncOpenAI client itself is
        # a lazily-created module-level singleton shared across instances.
        if not os.getenv("OPENAI_API_KEY", ""):
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")

        self.context = None
        self.doc_type = None  # 'exam_paper', 'exercise_set', 'research_article', 'lecture_notes' or None
        self.conversation_history = []

        # System prompt for the AI tutor
        self.system_prompt = r"""You are an intelligent and patient AI tutor. Your role is to help students learn and understand their study material (lecture notes, past exam/test papers, tutorial/exercise/homework question sheets, or academic research articles) effectively.

        Key behaviors:
        - Be encouraging and supportive
        - Break down complex concepts into digestible parts
        - Use examples and analogies to explain difficult topics
        - Ask follow-up questions to check for understanding
        - Provide practice questions and exercises when appropriate
        - Adapt your teaching style to the student's needs
        - Always base your responses on the provided study material context (lecture notes, an exam paper, a tutorial/homework question sheet, or a research article)
        - If asked about something not in the material, acknowledge this and provide general guidance
        - Use emojis sparingly but appropriately to maintain engagement

        CRITICAL FORMATTING REQUIREMENTS FOR CHAT:
        • **MATHEMATICAL NOTATION:** Write mathematics in proper LaTeX so it renders beautifully:
          * Use \(...\) for inline math and \[...\] for display math
          * Do NOT use $ or $$ delimiters (they are disabled)
          * Every subscript and superscript MUST have braces: \(P_{t}\), \(\sigma^{2}\), \(\hat{\mu}_{12}\)
          * Escape percent signs as \% and ampersands as \& inside math - a bare % or & breaks the rendering
          * Currency: use the real symbols with amounts (£1,000, $500, €250), NOT the words pounds/dollars/euros. £ and € may be written directly inside math; the dollar sign inside math MUST be escaped as \$ (a raw $ inside math breaks the rendering)
          * Use \begin{align*} with \\[6pt] line spacing between lines for multi-step calculations
          * Do NOT use LaTeX spacing commands (\;, \!, \,, \:) or an overline/vinculum
        • **CITATIONS:** The study material contains markers like "--- Page 4 ---" or "--- Slide 12 ---". When your answer draws on a specific part of the notes, cite it naturally at the end of the relevant sentence, e.g. "(see Slide 12)" or "(Pages 4-5)". Only cite page or slide numbers that actually appear in the markers - NEVER invent them. Do not quote the markers themselves.
        • Use markdown formatting (**bold**, *italic*, `code`, bullet points with hyphens)
        • For bold and italic, use ONLY asterisk syntax (**bold**, *italic*) - NEVER underscore syntax (__bold__, _italic_)
        • **HEADING FORMATTING**: Main headings in your responses must be in BLOCK CAPITALS and formatted with both bold and italic markdown: ***LIKE THIS***.
        Sub-headings should be in BLOCK CAPITALS with bold markdown: **LIKE THIS**.
        Keywords should be in italics *Like this*.
        • NEVER use HTML tags or hash (#) headings in your responses - only the markdown formats above
        • Always respond in plain text with markdown formatting - never return HTML, JSON, or other markup languages unless explicitly requested

        Remember: Your goal is to enhance learning, not just provide answers.

        ESSENTIAL: THIS IS A STUDY AND REVISION TOOL. NEVER ALLOW STUDENTS TO CHEAT BY PROVIDING EXTENSIVE ESSAY TYPE ANSWERS.

        ESSENTIAL: THIS TOOL IS ONLY TO BE USED FOR THE PURPOSES OF HELPING STUDENTS AT QUEEN'S UNIVERSITY BELFAST (QUB) TO STUDY AND REVISE FOR THEIR FINANCE COURSES. IT IS NOT TO BE USED FOR ANY OTHER PURPOSE.

        """

    # All AI calls go through _make_async_openai_fallback_call or
    # _make_async_openai_streaming_call: MODEL_PRIMARY first, MODEL_FALLBACK
    # as the fallback.

    def _get_truncated_context(self, limit=MAX_CONTEXT_CHARS):
        """Return the document context within `limit` characters.

        Documents over the limit keep their head and tail with an explicit
        elision note, so later pages stay visible and the model knows
        material was omitted (rather than silently losing the tail).
        """
        if len(self.context) <= limit:
            return self.context
        head = self.context[:int(limit * 0.8)]
        tail = self.context[-int(limit * 0.15):]
        return (head
                + "\n\n[NOTE: a middle section of the document was omitted here "
                "because the document is too long to include in full.]\n\n"
                + tail)

    def _select_relevant_context(self, query, limit=CHAT_CONTEXT_CHARS):
        """Context for chat: the whole document when it fits, otherwise the
        pages/slides most relevant to the query.

        Pages are scored by occurrences of the query's content words, then
        packed in document order (markers intact, so citations stay valid).
        The first page is always kept as an anchor for module/topic framing.
        Falls back to head+tail truncation when there are no page markers or
        the query has no usable content words.
        """
        if len(self.context) <= limit:
            return self.context
        chunks = [c for c in _PAGE_MARKER_SPLIT.split(self.context) if c.strip()]
        if len(chunks) < 2:
            return self._get_truncated_context(limit)
        terms = set(re.findall(r'[a-z0-9]{3,}', (query or '').lower())) - _STOPWORDS
        if not terms:
            return self._get_truncated_context(limit)

        scored = []
        for idx, chunk in enumerate(chunks):
            low = chunk.lower()
            scored.append((sum(low.count(t) for t in terms), idx))

        keep = {0}
        used = len(chunks[0])
        for score, idx in sorted(scored, key=lambda s: (-s[0], s[1])):
            if score <= 0 or idx in keep:
                continue
            if used + len(chunks[idx]) > limit:
                continue
            keep.add(idx)
            used += len(chunks[idx])

        if len(keep) < 2:
            # Nothing scored: the question doesn't match page text
            return self._get_truncated_context(limit)
        logging.info(f"CHAT CONTEXT: selected {len(keep)}/{len(chunks)} pages ({used} chars) for query")
        return ("[NOTE: only the pages of the study material most relevant to the "
                "student's question are included below; the document has more pages "
                "that are omitted here.]\n\n"
                + "\n".join(chunks[i] for i in sorted(keep)))

    def _build_feature_messages(self, persona, prompt):
        """Build the standard system+user message pair for a document feature."""
        truncated_context = self._get_truncated_context()
        return [
            {"role": "system", "content": f"{persona}\n\n{self._material_guidance()}{self._material_label()}:\n{truncated_context}"},
            {"role": "user", "content": prompt},
        ]

    def _build_api_args(self, messages, model, temperature, max_tokens,
                        response_format=None, reasoning_effort=None, stream=False):
        """Build chat.completions arguments, handling model capability differences."""
        if model in NO_SYSTEM_MESSAGE_MODELS:
            # These models don't support system messages - combine all messages
            # into a single user message.
            combined_content = ""
            for message in messages:
                if message["role"] == "system":
                    combined_content += f"System: {message['content']}\n\n"
                else:
                    combined_content += f"{message['content']}\n\n"
            api_args = {
                "model": model,
                "messages": [{"role": "user", "content": combined_content.strip()}],
                "max_completion_tokens": max_tokens,
                # NOTE: temperature is deliberately omitted here - the gpt-5.6-*
                # reasoning models reject non-default temperature values.
            }
            if reasoning_effort:
                api_args["reasoning_effort"] = reasoning_effort
        else:
            # Regular OpenAI models (gpt-4o-mini, etc.)
            api_args = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        if response_format:
            api_args["response_format"] = response_format
        if stream:
            api_args["stream"] = True
        return api_args

    async def _make_async_openai_fallback_call(self, messages, model=MODEL_FALLBACK, temperature=0.7,
                                               max_tokens=20000, response_format=None, timeout=50,
                                               reasoning_effort=None):

        client = _get_async_openai_client()
        api_args = self._build_api_args(messages, model, temperature, max_tokens,
                                        response_format=response_format,
                                        reasoning_effort=reasoning_effort)

        # The shared SDK client is configured with max_retries=2, so transient
        # connection/rate-limit/5xx errors are already retried with backoff
        # inside the SDK. Keep exactly one application-level retry, for
        # overall-call timeouts only.
        max_retries = 1
        for attempt in range(max_retries + 1):
            try:
                response = await asyncio.wait_for(
                    client.chat.completions.create(**api_args),
                    timeout=timeout
                )

                content = response.choices[0].message.content
                logging.debug(f"Async OpenAI response: finish_reason={response.choices[0].finish_reason}, content_length={len(content) if content else 0}")

                if not content or not content.strip():
                    finish_reason = response.choices[0].finish_reason
                    logging.error(f"Async OpenAI returned an empty response. Finish reason: {finish_reason}")
                    if finish_reason == "content_filter":
                        raise ValueError("Content was filtered by AI safety systems. Please try generating a different question.")
                    elif finish_reason == "length":
                        raise ValueError("Response was truncated due to length limits. Please try again.")
                    else:
                        raise ValueError("Async OpenAI service returned an empty response.")

                return content

            except asyncio.TimeoutError:
                logging.error(f"Async OpenAI call to {model} timed out after {timeout}s (attempt {attempt + 1}/{max_retries + 1})")
                if attempt < max_retries:
                    await asyncio.sleep(1)
                    continue
                raise

            # Classification is by exception TYPE (never by substring matching
            # on str(e)). The original exception is always logged and re-raised
            # so callers can decide on the user-facing message.
            except (openai.APIConnectionError, openai.RateLimitError) as e:
                # Retryable classes (APITimeoutError subclasses
                # APIConnectionError) - the SDK has already retried these.
                logging.error(f"Async OpenAI call to {model} failed with retryable error after SDK retries: {type(e).__name__} - {e}")
                raise
            except openai.APIStatusError as e:
                if e.status_code >= 500:
                    # Retryable server error - already retried by the SDK.
                    logging.error(f"Async OpenAI call to {model} failed with server error {e.status_code} after SDK retries: {type(e).__name__} - {e}")
                else:
                    # Client error (4xx) - not retryable, fail fast.
                    logging.error(f"Async OpenAI call to {model} failed with non-retryable API error {e.status_code}: {type(e).__name__} - {e}")
                raise
            except Exception as e:
                # Anything else (parsing errors, empty responses, etc.) - fail fast.
                logging.error(f"Async OpenAI call to {model} failed with non-retryable error: {type(e).__name__} - {e}")
                raise

    async def _make_async_openai_streaming_call(self, messages, model=MODEL_FALLBACK, temperature=0.7,
                                                max_tokens=20000, timeout=60, reasoning_effort=None,
                                                response_format=None):
        """Streaming variant of _make_async_openai_fallback_call. Yields text chunks.

        A leading markdown code fence (``` or ```lang) is stripped from the
        start of the stream: output is buffered until the first newline when the
        stream begins with a backtick, so consumers never see the fence.
        """
        client = _get_async_openai_client()
        api_args = self._build_api_args(messages, model, temperature, max_tokens,
                                        response_format=response_format,
                                        reasoning_effort=reasoning_effort, stream=True)

        stream = await asyncio.wait_for(
            client.chat.completions.create(**api_args),
            timeout=timeout
        )

        lead_buffer = ""
        lead_done = False
        async for chunk in stream:
            if not (chunk.choices and chunk.choices[0].delta.content):
                continue
            text = chunk.choices[0].delta.content
            if lead_done:
                yield text
                continue

            lead_buffer += text
            probe = lead_buffer.lstrip()
            if not probe:
                continue  # only whitespace so far, keep buffering
            if not probe.startswith("`"):
                lead_done = True
                yield lead_buffer
                lead_buffer = ""
            elif "\n" in probe:
                first_line, rest = probe.split("\n", 1)
                if re.fullmatch(r'```[A-Za-z0-9_+\-]*[ \t\r]*', first_line):
                    out = rest  # drop the opening fence line
                else:
                    out = lead_buffer  # backtick but not a fence - emit as-is
                lead_done = True
                if out:
                    yield out
                lead_buffer = ""
            elif len(probe) > 40:
                # Too long to be a fence line - emit as-is
                lead_done = True
                yield lead_buffer
                lead_buffer = ""

        if not lead_done and lead_buffer:
            # The whole stream fit in the buffer (or ended mid-fence-line)
            flushed = _strip_code_fences(lead_buffer)
            if flushed:
                yield flushed

    async def _stream_with_fallback(self, stream_factory, label, failure_message):
        """Run a primary-model stream with a fallback to the secondary model.

        The fallback is only attempted if the primary stream failed BEFORE any
        chunk was emitted. If output has already reached the user, restarting
        from scratch would show a truncated answer followed by a full second
        answer, so instead we log the error and emit a short interruption
        marker.
        """
        emitted = False
        primary_error = None
        try:
            async for chunk in stream_factory(MODEL_PRIMARY):
                emitted = True
                yield chunk
            return
        except Exception as e:
            primary_error = e
            if emitted:
                logging.error(f"{label}: {MODEL_PRIMARY} failed mid-stream after output was emitted: {e}")
                yield "\n\n[Connection interrupted - please ask me to continue]"
                return
            logging.error(f"{label}: {MODEL_PRIMARY} failed before emitting output: {e}; falling back to {MODEL_FALLBACK}")

        try:
            async for chunk in stream_factory(MODEL_FALLBACK):
                emitted = True
                yield chunk
        except Exception as e:
            if emitted:
                logging.error(f"{label}: {MODEL_FALLBACK} failed mid-stream after output was emitted: {e}")
                yield "\n\n[Connection interrupted - please ask me to continue]"
                return
            logging.error(f"{label}: both models failed: {primary_error} | {e}")
            yield failure_message

    async def close_async_clients(self):
        """Compatibility no-op retained for app.py finally blocks.

        The AsyncOpenAI client is a module-level singleton shared across
        requests and TutorAI instances, so it must NOT be closed per-request.
        """
        logging.debug("close_async_clients() called - shared client left open (no-op)")

    async def _summarize_for_infographic(self, char_limit=8000):
        """Condense the ENTIRE lecture into a revision brief that fits the
        images API prompt budget, so the infographic covers the whole lecture
        rather than just whatever survives a head/tail truncation."""
        full_context = self._get_truncated_context()
        summary_prompt = rf"""Condense the following uploaded university material (lecture slides, notes, a test/exam paper, a tutorial/exercise/homework question sheet, or an academic research article) into a revision brief that will be handed to an image-generation model to design a one-page revision infographic.

STRICT REQUIREMENTS:
- The brief MUST be under {char_limit} characters in total.
- Cover the WHOLE document from beginning to end — every major topic must appear; do not stop early or skip later sections.
- Start with a single title line naming the subject of the material.
- Then give 4-6 clearly titled sections (fewer, broader sections are better than many small ones). In each section give AT MOST 3 bullet points, each a crisp phrase of no more than 9 words that fits on ONE line of a poster - only the single most important concepts, definitions and takeaways. This is a visual poster, not notes: leave out detail.
- After each section's bullets add ONE line starting "VISUAL:" choosing ONE of: (a) "VISUAL: diagram - ..." a clean explanatory diagram (timeline, flowchart, labelled graph, relationship map, comparison) - the DEFAULT for any quantitative, structural or process idea; (b) "VISUAL: vignette - ..." a SMALL photorealistic vignette used only where a photo adds visual appeal to a concept that has no natural diagram; or (c) "VISUAL: formula-led" when the section's formula is the visual and nothing else is needed. Use at most 2 vignettes across the WHOLE brief, each of a clearly different subject (never two of the same scene, and never "a person looking at trading screens"). Vignette subjects must be concrete and physical (a bond certificate, a factory, a shopfront, coins, a signed contract) rather than screens, charts or documents with text. Never suggest icons or clip-art. Diagrams must only show relationships or structures described in the material, never invented data.
- If the material contains equations, include the main equations a student must learn for the exam INSIDE the section they belong to (not in a separate formulas section), each on its own line starting "FORMULA:" in GENERAL symbolic form written in LaTeX math notation, followed by " KEY: " and a few-word plain-text note of what each symbol means (e.g. "FORMULA: F = P(1 + r)^t KEY: F future value, P present value, r rate, t years"). Include at most 6 formulas across the whole brief - the ones that matter most - and no more than 2 per section. A FORMULA line does not count towards the 3-bullet limit.
- FORMULA NOTATION (LaTeX, so the poster can typeset real mathematics): use \frac{{numerator}}{{denominator}} for EVERY division (never a "/" slash); _{{ }} and ^{{ }} for subscripts and superscripts (r_{{i}}, \sigma^{{2}}, P_{{0}}); \bar{{r}} for a mean, \hat{{x}} for an estimate; Greek letters as commands (\sigma, \rho, \beta, \mu); \sum for summation (with limits if the material shows them, e.g. \sum_{{i=1}}^{{n}}); \sqrt{{ }} for roots; \times for multiplication. NEVER spell a symbol as a word (not "rbar", "sigma", "sqrt", "sum"). Example: "FORMULA: \sigma^{{2}} = \frac{{\sum (r_{{i}} - \bar{{r}})^{{2}}}}{{n - 1}} KEY: r_i return, r-bar mean return, n observations". The KEY part is plain words, and it must list EVERY symbol that appears in the formula, each written as the symbol followed by its meaning (e.g. 'KEY: r_p portfolio return, r_f risk-free rate, sigma_p portfolio volatility') - never a bare list of meanings without the symbols.
- FORMULA ACCURACY: copy each formula from the material exactly. Be meticulous about brackets - what is inside versus outside each bracket, and which terms an exponent, root, sum or division applies to. Write explicit brackets and braces wherever the structure could be misread, e.g. "PV = \frac{{C}}{{(1 + r)^{{t}}}}" (the whole (1 + r) is raised to t, then divides C), not "PV = C / 1 + r^t". Never rearrange, simplify or merge formulas.
- Plain text only: no markdown symbols, no page/slide citations, no commentary about these instructions — output the brief and nothing else.

THIS IS A REVISION RECORD, NOT A WORKSHEET (MOST IMPORTANT):
- The poster is a record of the key concepts and general formulas a student needs for the exam. Show ideas and general equations, NEVER specific numerical worked examples.
- Do NOT include worked examples, practice questions, exercise figures or their answers. For instance, show "F = P(1 + r)^t" but do NOT show "when P = 100, r = 5% and t = 2 the answer is 110.25". If the material only presents a formula through a numerical example, restate it in general symbolic form.
- Do NOT calculate anything yourself and do NOT invent formulas, facts or examples that are not in the material.
- Use ONLY information that actually appears in the uploaded material - no outside knowledge.
- VISUAL suggestions must also come only from what the material discusses - do not suggest imagery for topics it does not cover.

{self._infographic_material_note()}UPLOADED MATERIAL:
{full_context}"""
        messages = [{"role": "user", "content": summary_prompt}]

        try:
            logging.info(f"INFOGRAPHIC: Summarizing lecture with {MODEL_PRIMARY} for the image prompt...")
            summary = await self._make_async_openai_fallback_call(
                messages=messages, model=MODEL_PRIMARY, temperature=0.2,
                max_tokens=4000, timeout=90
            )
        except Exception as primary_error:
            logging.error(f"INFOGRAPHIC: {MODEL_PRIMARY} summarization failed: {primary_error}; trying {MODEL_FALLBACK}")
            summary = await self._make_async_openai_fallback_call(
                messages=messages, model=MODEL_FALLBACK, temperature=0.2,
                max_tokens=4000, timeout=90
            )

        summary = summary.strip()
        if len(summary) > char_limit:
            logging.warning(f"INFOGRAPHIC: Summary overshot the {char_limit}-char budget ({len(summary)} chars); trimming")
            summary = summary[:char_limit]
        logging.info(f"INFOGRAPHIC: Lecture condensed to {len(summary)} chars for the image prompt")
        return summary

    def _infographic_material_note(self):
        """Material-type note for the infographic brief. Question sets become a record of the
        concepts and formulas the questions require, never the questions or their figures."""
        if self.is_research_article():
            return self._RESEARCH_GUIDANCE
        if self.is_question_set():
            return (
                f"MATERIAL TYPE: The uploaded document is {self._material_description()}. Do NOT list the "
                "questions or their figures. Instead, extract the concepts, definitions and general "
                "formulas a student would need to know to answer them, and present those as the "
                "revision content.\n\n"
            )
        return ""

    async def generate_infographic_async(self):
        """Generate a one-page revision-guide infographic for the current
        document using the OpenAI images API. Returns a base64-encoded PNG."""
        if not self.context:
            raise ValueError("No document context set for infographic generation")

        # The images API accepts a far smaller prompt than the chat models, so
        # first condense the WHOLE lecture into that budget with a chat-model
        # summarization pass (a plain truncation would drop later sections).
        try:
            notes_brief = await self._summarize_for_infographic(char_limit=5000)
        except Exception as e:
            logging.error(f"INFOGRAPHIC: Summarization failed, falling back to head/tail truncation: {e}")
            notes_brief = self._get_truncated_context(limit=5000)

        prompt = (
            "Design a beautiful, modern one-page revision poster for the university "
            "revision brief below. It must look like a premium editorial infographic, "
            "not a page of notes.\n\n"
            "VISUAL STYLE:\n"
            "- Diagram-led: each section's main visual is what its VISUAL line asks "
            "for. A 'diagram' is a clean explanatory diagram (flowchart, timeline, "
            "labelled graph, relationship map, comparison) drawn in the page's style "
            "(thin red/grey strokes, white boxes, sans-serif labels) and given room "
            "to breathe. A 'formula-led' section has no picture: its typeset "
            "formula callout is the visual, with the extra space left white.\n"
            "- PHOTOS ARE SMALL ACCENTS, NOT FEATURES: a 'vignette' is a small "
            "photorealistic image - no more than about one fifth of its card's "
            "area, roughly the height of three bullet lines - with softly rounded "
            "corners, placed in a corner of the card or beside the formula so the "
            "text, diagram and formula remain the dominant elements. Never let a "
            "photo fill half a card or the full height of a card. At most 2 "
            "vignettes on the whole page, each a clearly different subject; never "
            "repeat a scene, never show people looking at trading screens, and "
            "never show screens, charts or documents with text inside a photo. If "
            "no vignette is requested for a section, do not add one.\n"
            "- No icons, clip-art or cartoon illustrations; small icons may only "
            "appear as subtle bullet markers.\n"
            "- Hero image: one small photorealistic vignette beside the title "
            "(about one fifth of the page width, no taller than the title block), "
            "of a subject different from every other vignette on the page.\n"
            "- Diagrams must depict only the structures and relationships described in "
            "the brief; graph axes may be labelled but show NO invented numbers.\n"
            "- CLEAN STYLE: the page background must be pure white (no red, pink or "
            "any colour tint, no gradient, no textured or coloured backdrop). Section "
            "cards are white with a hairline light-grey border and a very soft "
            "shadow; card interiors stay white.\n"
            "- ACCENT COLOUR: a single deep red (like #D6000D) used ONLY as thin "
            "strokes and text - section headings, a thin rule under each heading, "
            "arrows and connector lines in diagrams, the left border of formula "
            "callouts, and small numbered badges. Red must NEVER be used as a fill "
            "for large areas: no red or pink card backgrounds, panels, bands, boxes "
            "behind formulas or tinted washes. Diagram nodes are white or pale grey "
            "boxes with a thin red or grey outline and charcoal text, not solid "
            "coloured blocks. All other colour comes only from the photographs.\n"
            "- LAYOUT RHYTHM: a clean 2-column grid of equal-width cards (a full-width "
            "card is allowed for a wide diagram). Every card has the SAME anatomy: a "
            "small red number badge, the heading, a thin red rule, then bullets on the "
            "left and the visual on the right, with any formula callout beneath the "
            "bullets. Equal card padding, equal gutters between cards, and a clear "
            "margin on all four page edges (nothing touches the edge, including the "
            "bottom). Balance card heights so no card is cramped or half-empty.\n"
            "- Generous white space, aligned grid, rounded corners, clear visual "
            "hierarchy - the feel of a premium magazine spread.\n\n"
            "TEXT RULES:\n"
            "- Keep text minimal: one bold title, one short heading per section, and at "
            "most 3 short bullet phrases per section. No paragraphs, no small print.\n"
            "- TYPOGRAPHY: one clean, modern geometric sans-serif family for ALL "
            "text (in the style of Inter, Helvetica Neue or Roboto) - no serif, "
            "script, handwritten, condensed, decorative or display fonts for text, "
            "and never mix text families. The ONLY exception is mathematics: "
            "formulas are set in a classic LaTeX-style math font (Computer Modern / "
            "Latin Modern look: serif, italic variables, upright operators). Regular weight for body text, semi-bold for section "
            "headings, bold only for the main title. Consistent sizes: one size for "
            "all section headings, one for all bullet text, one for formula "
            "callouts. Dark charcoal text on white, left-aligned, generous line "
            "spacing, no drop shadows, outlines, gradients or effects on text.\n"
            "- Every word must be spelled correctly, crisply rendered and readable.\n"
            "- Include every section of the brief, but let visuals carry the meaning "
            "wherever they can replace words.\n\n"
            "FORMULAS: Lines marked FORMULA belong inside the section they appear "
            "in - do NOT gather them into a separate formulas panel. Within each "
            "section, show its formula(s) as a callout: a white or very pale grey box "
            "with a thin red left border, the formula large in charcoal, with the "
            "short symbol KEY in small grey sans-serif text beneath.\n"
            "MATHS TYPESETTING: each FORMULA in the brief is written in LaTeX "
            "notation. Render it as properly typeset mathematics, exactly as a "
            "LaTeX document would print it - in a Computer Modern / Latin Modern "
            "style math font, with italic single-letter variables and upright "
            "operators and numbers. Specifically:\n"
            "- Every division (frac) is a STACKED fraction: numerator on the top "
            "line, a horizontal fraction bar, denominator on the bottom line - "
            "never an inline slash.\n"
            "- Subscripts and superscripts are true small raised/lowered glyphs.\n"
            "- A bar over a letter for a mean (r with a bar above it), a hat for an "
            "estimate; Greek letters as their proper glyphs; summation as a large "
            "sigma sign with any limits above and below; square roots with a "
            "radical sign spanning the whole radicand.\n"
            "- NEVER print LaTeX source (no backslashes, braces or command names) "
            "and NEVER spell symbols as words (no 'rbar', 'sigma', 'sqrt', 'sum').\n"
            "- Reproduce every bracket exactly, keeping the same terms inside and "
            "outside each bracket as in the brief, and make it visually clear "
            "which terms an exponent, fraction bar, root or summation applies to. "
            "Do not drop, add, move or nest brackets, and do not alter, merge or "
            "invent symbols, subscripts or exponents.\n\n"
            "STRICT CONTENT RULES: This is a revision record of key concepts and "
            "general formulas. Show ONLY information contained in the brief. Do not "
            "add facts, formulas, examples or explanations from outside it. Show NO "
            "numerical worked examples, practice questions or calculated answers - "
            "general equations only, never numbers substituted into them.\n\n"
            f"REVISION BRIEF:\n{notes_brief}"
        )

        client = _get_async_openai_client()
        logging.info(f"INFOGRAPHIC: Requesting image generation ({self.INFOGRAPHIC_MODEL}, 1024x1536, high quality)")
        response = await client.images.generate(
            model=self.INFOGRAPHIC_MODEL,
            prompt=prompt,
            n=1,
            size="1024x1536",
            quality="high",
            timeout=300,
        )
        image_b64 = response.data[0].b64_json if response.data else None
        if not image_b64:
            raise ValueError("Image generation returned no image data")
        logging.info(f"INFOGRAPHIC: Image received ({len(image_b64)} base64 chars)")

        # ---- Check phase: inspect the image against the brief and correct it ----
        image_b64 = await self._check_and_correct_infographic(image_b64, notes_brief)
        return image_b64

    INFOGRAPHIC_MODEL = "gpt-image-2.5-sunburst"
    INFOGRAPHIC_MAX_FIX_ROUNDS = 2

    async def _review_infographic(self, image_b64, notes_brief):
        """Ask a vision-capable chat model to compare the rendered infographic with the
        brief. Returns a dict {"ok": bool, "issues": [{"location", "problem", "correction"}]}.
        Raises on API failure (caller decides whether to proceed without a check)."""
        review_prompt = f"""You are proofreading a one-page revision infographic that was generated from the REVISION BRIEF below. Inspect the image carefully and report every problem that would mislead a student or that breaks the required style.

CHECK, IN THIS ORDER OF IMPORTANCE:
1. FORMULAS: every formula shown must be the brief's LaTeX formula rendered as properly typeset mathematics: the same symbols, subscripts and exponents, and the same bracket placement (the same terms inside and outside each bracket, and the correct scope of every exponent, fraction or root). ALSO flag: any division shown inline with a slash instead of a stacked fraction (numerator over a horizontal bar over denominator); any symbol spelled as a word (e.g. 'rbar', 'sigma', 'sqrt') instead of the proper glyph (bar over the letter, Greek letter, radical sign); any visible LaTeX source (backslashes, braces, command names); formulas set in a plain sans-serif font instead of a LaTeX-style (Computer Modern) math font. Any deviation is an issue. In 'correction', give the exact formula from the brief and say how it must be typeset.
2. TEXT ACCURACY: misspelled, garbled, truncated or unreadable words; headings or bullets that say something the brief does not.
3. CONTENT RULES: any numerical worked example, practice question, substituted numbers or calculated answer (none are allowed); any fact, formula or example not in the brief; any section of the brief missing entirely.
4. STYLE: a red / pink / coloured page background, card background, band or colour wash (the page and cards must be white; red may appear only as headings, thin rules, arrows, outlines and small badges - never as a fill behind text or formulas); any accent colour other than red (e.g. navy or blue fills); serif, script or decorative fonts; flat icons or clip-art used as a section's main visual; a photo that dominates its card (larger than about one fifth of the card, or full card height); more than 2 photos on the page besides the small hero image, or two photos of the same kind of scene (e.g. people at trading screens twice); a photo containing screens, charts or readable text; a symbol key that omits the symbols (e.g. 'portfolio return, risk-free rate' without r_p, r_f).

OUTPUT: respond with ONLY a JSON object, no other text:
{{"ok": true}} if there are no issues, otherwise
{{"ok": false, "issues": [{{"location": "<section heading or area of the page>", "problem": "<what is wrong>", "correction": "<exactly what it should show instead>"}}]}}
List at most 8 issues, most important first. Be precise and literal - do not invent problems, and do not report stylistic preferences beyond rule 4.

REVISION BRIEF:
{notes_brief}"""
        client = _get_async_openai_client()
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": review_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}", "detail": "high"}},
            ],
        }]
        last_error = None
        for model in (MODEL_PRIMARY, MODEL_FALLBACK):
            try:
                api_args = {
                    "model": model,
                    "messages": messages,
                    "max_completion_tokens": 4000,
                    "response_format": {"type": "json_object"},
                }
                response = await asyncio.wait_for(client.chat.completions.create(**api_args), timeout=120)
                content = (response.choices[0].message.content or "").strip()
                result = json.loads(content)
                if not isinstance(result, dict):
                    raise ValueError("Review response was not a JSON object")
                issues = result.get("issues") or []
                result = {"ok": bool(result.get("ok", not issues)) and not issues, "issues": issues[:8]}
                logging.info(f"INFOGRAPHIC CHECK ({model}): ok={result['ok']}, {len(result['issues'])} issue(s)")
                return result
            except Exception as e:
                last_error = e
                logging.error(f"INFOGRAPHIC CHECK: review with {model} failed: {e}")
        raise last_error if last_error else RuntimeError("Infographic review failed")

    async def _fix_infographic(self, image_b64, issues, notes_brief):
        """Use the image edit endpoint to correct the listed issues while keeping the
        rest of the poster unchanged. Returns the corrected base64 PNG."""
        import base64
        corrections = "\n".join(
            f"{i + 1}. {issue.get('location', 'page')}: {issue.get('problem', '')} -> "
            f"Correct to: {issue.get('correction', '')}"
            for i, issue in enumerate(issues)
        )
        fix_prompt = (
            "Edit this revision infographic. Keep the overall layout, section order, "
            "photographs, diagrams and all correct text exactly as they are. Apply ONLY "
            "the corrections listed below, redrawing just the affected areas.\n\n"
            f"CORRECTIONS:\n{corrections}\n\n"
            "RULES WHILE EDITING: formulas must be typeset with complete accuracy - "
            "exactly the symbols, subscripts, exponents and bracket placement given in "
            "the correction, with nothing dropped, added or rearranged - as proper "
            "LaTeX-style mathematics (Computer Modern look): stacked fractions with a "
            "horizontal bar, true sub/superscripts, bars and hats over letters, Greek "
            "glyphs, a large sigma for sums; never a slash for division, never symbols "
            "spelled as words, never visible LaTeX source. Text must be "
            "spelled correctly and fully legible. Show no numerical worked examples or "
            "calculated answers. The page and all cards must stay pure white with no "
            "red or coloured tint or fills; the only accent is deep red used for "
            "headings, thin rules, arrows, outlines and small badges, and formula "
            "boxes are white with a thin red left border. Keep one clean sans-serif "
            "font family. Photos may only be small rounded vignettes (about one fifth "
            "of a card), at most 2 on the page plus a small hero image, all of "
            "different subjects, with no screens, charts or text inside them - "
            "shrink, replace or remove a photo if a correction asks for it and give "
            "the space to the diagram, formula or white space. Do not add any "
            "content that is not in the brief.\n\n"
            f"REVISION BRIEF (source of truth):\n{notes_brief}"
        )
        client = _get_async_openai_client()
        image_bytes = base64.b64decode(image_b64)
        logging.info(f"INFOGRAPHIC FIX: requesting edit for {len(issues)} issue(s)")
        response = await client.images.edit(
            model=self.INFOGRAPHIC_MODEL,
            image=("infographic.png", image_bytes, "image/png"),
            prompt=fix_prompt,
            input_fidelity="high",
            n=1,
            size="1024x1536",
            quality="high",
            timeout=300,
        )
        fixed_b64 = response.data[0].b64_json if response.data else None
        if not fixed_b64:
            raise ValueError("Image edit returned no image data")
        logging.info(f"INFOGRAPHIC FIX: corrected image received ({len(fixed_b64)} base64 chars)")
        return fixed_b64

    async def _check_and_correct_infographic(self, image_b64, notes_brief):
        """Review the generated infographic against the brief and correct it, up to
        INFOGRAPHIC_MAX_FIX_ROUNDS times. Never fails the whole generation: if the
        review or the edit errors, the best image so far is returned."""
        current = image_b64
        for round_no in range(1, self.INFOGRAPHIC_MAX_FIX_ROUNDS + 1):
            try:
                review = await self._review_infographic(current, notes_brief)
            except Exception as e:
                logging.error(f"INFOGRAPHIC CHECK: round {round_no} review unavailable, returning current image: {e}")
                return current
            if review["ok"]:
                logging.info(f"INFOGRAPHIC CHECK: passed on round {round_no}")
                return current
            for issue in review["issues"]:
                logging.info(f"INFOGRAPHIC CHECK: issue - {issue.get('location')}: {issue.get('problem')}")
            try:
                current = await self._fix_infographic(current, review["issues"], notes_brief)
            except Exception as e:
                logging.error(f"INFOGRAPHIC FIX: round {round_no} edit failed, returning current image: {e}")
                return current
        # Final verification after the last fix round (log only)
        try:
            final = await self._review_infographic(current, notes_brief)
            if final["ok"]:
                logging.info("INFOGRAPHIC CHECK: passed after corrections")
            else:
                logging.warning(f"INFOGRAPHIC CHECK: {len(final['issues'])} issue(s) remain after "
                                f"{self.INFOGRAPHIC_MAX_FIX_ROUNDS} fix round(s); returning best image")
        except Exception as e:
            logging.error(f"INFOGRAPHIC CHECK: final review unavailable: {e}")
        return current

    def set_context(self, pdf_content, doc_type=None):

        self.context = pdf_content
        self.doc_type = doc_type  # 'exam_paper', 'exercise_set', 'research_article', 'lecture_notes' or None
        self.conversation_history = []  # Reset conversation when new context is set

    # Cheap textual cues used only when no LLM classification has been stored
    # for the session. Exam cues: marks, time limits, "Answer ALL questions"...
    _EXAM_HINT_RE = re.compile(
        r"(\[\s*\d+\s*marks?\s*\]|\(\s*\d+\s*marks?\s*\)|answer\s+all\s+questions|"
        r"answer\s+any\s+\w+\s+questions|time\s+allowed|\bquestion\s+\d+\b|"
        r"\bq\s*\d+\s*[.)]|total\s+marks|this\s+paper|examination|exam\s+paper)",
        re.I,
    )
    # Strong exam-only cues that override exercise cues.
    _EXAM_STRONG_RE = re.compile(
        r"(time\s+allowed|answer\s+all\s+questions|answer\s+any\s+\w+\s+questions|"
        r"examination|exam\s+paper|total\s+marks|invigilat)", re.I)
    # Tutorial / exercise / homework / problem-set cues.
    _EXERCISE_HINT_RE = re.compile(
        r"(\btutorial\b|\bexercises?\b|\bhomework\b|problem\s+(set|sheet)|worksheet|"
        r"\bassignment\b|\bseminar\b|practice\s+questions|self[- ]study\s+questions|"
        r"questions\s+for\s+discussion|\bworkshop\b)", re.I)

    # Academic research article cues.
    _RESEARCH_HINT_RE = re.compile(
        r"(\babstract\b|\bkeywords?\b|jel\s+classification|literature\s+review|"
        r"\bmethodology\b|\bet\s+al\.|\breferences\b|\bjournal\s+of\b|\bdoi\b|"
        r"\bhypothes[ie]s\b|\bwe\s+find\b|\bour\s+(results|findings|sample)\b|"
        r"\bworking\s+paper\b|robustness|\bempirical\b)", re.I)

    DOC_TYPES = ("exam_paper", "exercise_set", "research_article", "lecture_notes")

    def _classify_heuristically(self):
        """Keyword fallback for the document type; caches the result on the instance."""
        sample = self.context[:20000]
        exam_hits = len(self._EXAM_HINT_RE.findall(sample))
        strong_exam = len(self._EXAM_STRONG_RE.findall(sample))
        exercise_hits = len(self._EXERCISE_HINT_RE.findall(sample))
        research_hits = len(self._RESEARCH_HINT_RE.findall(sample))
        if research_hits >= 5 and strong_exam == 0:
            self.doc_type = "research_article"
        elif exercise_hits >= 2 and strong_exam == 0:
            self.doc_type = "exercise_set"
        elif exam_hits >= 4:
            self.doc_type = "exam_paper"
        else:
            self.doc_type = "lecture_notes"
        logging.info(f"DOC TYPE: heuristic classified document as {self.doc_type} "
                     f"({exam_hits} exam cues, {strong_exam} strong, {exercise_hits} exercise cues, "
                     f"{research_hits} research cues)")
        return self.doc_type

    def get_doc_type(self):
        """'exam_paper', 'exercise_set' or 'lecture_notes' (never None once context is set)."""
        if self.doc_type in self.DOC_TYPES:
            return self.doc_type
        if not self.context:
            return "lecture_notes"
        return self._classify_heuristically()

    def is_exam_paper(self):
        return self.get_doc_type() == "exam_paper"

    def is_exercise_set(self):
        """True for tutorial sheets, exercise sets, homework and problem sets."""
        return self.get_doc_type() == "exercise_set"

    def is_research_article(self):
        """True for academic journal articles, working papers and similar research papers."""
        return self.get_doc_type() == "research_article"

    def is_question_set(self):
        """True when the document is made up of questions (exam paper OR exercise set)."""
        return self.get_doc_type() in ("exam_paper", "exercise_set")

    def _material_label(self):
        return {"exam_paper": "Exam Paper",
                "exercise_set": "Tutorial / Exercise Questions",
                "research_article": "Research Article"}.get(self.get_doc_type(), "Lecture Notes")

    def _material_noun(self):
        """Short lower-case noun for use inside prompt sentences."""
        return {"exam_paper": "exam paper",
                "exercise_set": "exercise sheet",
                "research_article": "research article"}.get(self.get_doc_type(), "lecture notes")

    def _material_description(self):
        return {"exam_paper": "an EXAM / TEST PAPER",
                "exercise_set": "a TUTORIAL / EXERCISE / HOMEWORK QUESTION SHEET",
                "research_article": "an ACADEMIC RESEARCH ARTICLE"}.get(self.get_doc_type(), "")

    _RESEARCH_GUIDANCE = (
        "MATERIAL TYPE: The uploaded document is an ACADEMIC RESEARCH ARTICLE (journal article or "
        "working paper), not lecture notes. Organise your output around the paper's structure: the "
        "research question and motivation, the theory or hypotheses, the data and methodology, the "
        "key findings, the contribution to the literature, and any limitations or implications. "
        "Report the paper's results, figures and conclusions exactly as the authors state them - do "
        "NOT recompute, extrapolate or add findings that are not in the paper. Where useful, cite "
        "sections, tables or page markers that appear in the document.\n\n"
    )

    def _material_guidance(self):
        """Extra instructions injected into feature prompts for question sets and research articles."""
        if self.is_research_article():
            return self._RESEARCH_GUIDANCE
        if not self.is_question_set():
            return ""
        return (
            f"MATERIAL TYPE: The uploaded document is {self._material_description()} made up of questions, "
            "not lecture notes. Treat the questions as the syllabus: work from the topics, concepts, "
            "definitions, formulas and methods that the questions test, and organise your output "
            "around those topics. Reproduce question wording and any given figures exactly. Do NOT "
            "calculate, solve or invent answers to the questions, and only state an answer or "
            "result if the document itself shows it (e.g. a marking scheme, model answer or solutions "
            "section). You may refer to question numbers, marks and instructions that appear in the "
            "document.\n\n"
        )

    def _chat_material_guidance(self):
        """Material-specific guidance for the interactive chat (tutoring, not content generation)."""
        if self.is_research_article():
            return (
                "MATERIAL TYPE: The uploaded document is an ACADEMIC RESEARCH ARTICLE, not lecture notes. "
                "Help the student understand and critically evaluate it: clarify the research question, "
                "theory, data, methodology, findings and contribution, explain technical terms and "
                "econometric methods in plain language, and encourage them to question assumptions and "
                "limitations. Report the authors' results exactly as stated and never invent numbers. "
                "Cite the section, table or page marker you are drawing on.\n\n"
            )
        if not self.is_question_set():
            return ""
        return (
            f"MATERIAL TYPE: The uploaded document is {self._material_description()}, not lecture notes. "
            "When the student asks about a question from it, explain the concepts and method "
            "it tests and guide them through it step by step, checking their understanding, rather "
            "than handing over a complete model answer (this is especially important for homework "
            "and assessed work). Quote the question's given figures exactly and cite the question "
            "number (e.g. \"(Question 3)\") instead of a page or slide. If the document includes a "
            "marking scheme, model answer or solutions, you may use them; otherwise make clear that "
            "any final figure you reach is your own working, not an official answer.\n\n"
        )

    def clear_context(self):

        self.context = None
        self.conversation_history = []

    def _build_chat_messages(self, user_message):
        """Build the messages list for a chat API call."""
        if not self.context:
            return None

        truncated_context = self._select_relevant_context(user_message)
        system_content = f"{self.system_prompt}\n\n{self._chat_material_guidance()}{self._material_label()} Context:\n{truncated_context}"

        if self.conversation_history:
            recent_history = self.conversation_history[-10:]
            system_content += f"\n\nPrevious Conversation Context:\nThe following are the most recent messages from our ongoing conversation. Please build on this context when responding to the user's latest query. Use this history to maintain continuity and reference previous topics discussed:\n"
            for i, msg in enumerate(recent_history):
                role_label = "Student" if msg["role"] == "user" else "AI Tutor"
                system_content += f"{role_label}: {msg['content']}\n"
            system_content += "\nPlease use this conversation history to provide a contextually relevant response to the user's new message below."

        messages = [{"role": "system", "content": system_content}]
        messages.append({"role": "user", "content": user_message})
        return messages

    async def get_response_async(self, user_message):

        if not self.context:
            return "I need you to upload your lecture notes, an exam paper, a question sheet or a research article first before I can help you study! 📚"

        messages = self._build_chat_messages(user_message)

        try:
            # Use the primary model for general chat
            ai_response = await self._make_async_openai_fallback_call(
                messages=messages,
                model=MODEL_PRIMARY,
                temperature=0.7,
                max_tokens=15000,
                timeout=60
            )

            # Update conversation history
            self.conversation_history.append({"role": "user", "content": user_message})
            self.conversation_history.append({"role": "assistant", "content": ai_response})

            return ai_response

        except Exception as openai_error:
            # Fall back to the secondary model if the primary fails
            try:
                ai_response = await self._make_async_openai_fallback_call(
                    messages=messages,
                    model=MODEL_FALLBACK,
                    temperature=0.7,
                    max_tokens=15000,
                    timeout=60
                )

                # Update conversation history
                self.conversation_history.append({"role": "user", "content": user_message})
                self.conversation_history.append({"role": "assistant", "content": ai_response})

                return ai_response

            except Exception as nano_error:
                logging.error(f"Both {MODEL_PRIMARY} and {MODEL_FALLBACK} failed for general chat: {openai_error} | {nano_error}")
                return "I'm having trouble connecting to the AI service right now. This is likely a temporary issue. Please try again in a few moments."

    async def get_response_stream_async(self, user_message):
        """Stream chat response chunks via an async generator."""
        if not self.context:
            yield "I need you to upload your lecture notes, an exam paper, a question sheet or a research article first before I can help you study! 📚"
            return

        messages = self._build_chat_messages(user_message)

        async def _try_stream(model):
            full_response = ""
            async for text in self._make_async_openai_streaming_call(
                messages=messages, model=model, temperature=0.7, max_tokens=15000, timeout=60
            ):
                full_response += text
                yield text
            # Update conversation history with the complete response
            self.conversation_history.append({"role": "user", "content": user_message})
            self.conversation_history.append({"role": "assistant", "content": full_response})

        async for chunk in self._stream_with_fallback(
            _try_stream,
            "STREAM CHAT",
            "I'm having trouble connecting to the AI service right now. This is likely a temporary issue. Please try again in a few moments."
        ):
            yield chunk

    async def generate_cheat_sheet_async(self):

        if not self.context:
            return "No study material available to create sheet from."

        try:
            messages = self._build_feature_messages(
                "You are an expert at creating study aids and revision sheets from academic content.",
                self._get_summary_prompt()
            )

            logging.info(f"ASYNC SUMMARY: Context length: {len(self._get_truncated_context())} characters")

            # Try the primary model
            try:
                logging.info(f"ASYNC SUMMARY: Trying {MODEL_PRIMARY} for executive summary generation...")
                result = await self._make_async_openai_fallback_call(
                    messages=messages, model=MODEL_PRIMARY, temperature=0.2, max_tokens=15000, timeout=60
                )
                if result and result.strip():
                    logging.info(f"ASYNC SUMMARY: {MODEL_PRIMARY} succeeded")
                    return _normalize_study_formatting(_strip_code_fences(result))
                raise ValueError(f"{MODEL_PRIMARY} returned an empty response")
            except Exception as mini_error:
                logging.error(f"ASYNC SUMMARY: {MODEL_PRIMARY} failed: {mini_error}")

            # Fallback model
            try:
                logging.info(f"ASYNC SUMMARY: Trying {MODEL_FALLBACK} fallback...")
                result = await self._make_async_openai_fallback_call(
                    messages=messages, model=MODEL_FALLBACK, temperature=0.2, max_tokens=15000, timeout=60
                )
                logging.info(f"ASYNC SUMMARY: {MODEL_FALLBACK} fallback succeeded")
                return _normalize_study_formatting(_strip_code_fences(result))
            except Exception as nano_error:
                logging.error(f"ASYNC SUMMARY: All models failed: {nano_error}")
                return "I'm having trouble generating a summary right now. The document appears to be loaded successfully, but there may be a temporary issue with the AI service. Please try again in a moment or use the chat to ask specific questions about your document."

        except Exception as e:
            logging.error(f"ASYNC SUMMARY: Critical error in async summary generation: {e}")
            return f"Critical error in summary generation: {str(e)}"

    async def generate_cheat_sheet_stream_async(self):
        """Streaming version of generate_cheat_sheet_async. Yields text chunks."""
        if not self.context:
            yield "No study material available to create sheet from."
            return

        try:
            messages = self._build_feature_messages(
                "You are an expert at creating study aids and revision sheets from academic content.",
                self._get_summary_prompt()
            )

            def factory(model):
                return self._make_async_openai_streaming_call(
                    messages=messages, model=model, temperature=0.2, max_tokens=15000, timeout=90
                )

            async for chunk in _normalize_study_stream(self._stream_with_fallback(
                factory,
                "STREAM SUMMARY",
                "I'm having trouble generating a summary right now. Please try again in a moment."
            )):
                yield chunk
        except Exception as e:
            logging.error(f"STREAM SUMMARY: Critical error: {e}")
            yield f"Critical error in summary generation: {str(e)}"

    def _get_summary_prompt(self):
        """Return the summary/cheat sheet prompt text (single source for streaming and non-streaming)."""
        return r"""

Create an executive summary from the study material I provide (lecture notes, an exam/test paper, a tutorial/exercise/homework question sheet, or a research article - for any set of questions, summarise the topics and concepts the questions test, never the answers; for a research article, summarise its question, method, findings and contribution). Your output should begin with a short overview, followed by a selective bullet-point revision sheet covering the most important ideas - this is a quick-reference revision aid, not exhaustive notes.

### CRITICAL FORMATTING REQUIREMENTS - FOLLOW EXACTLY
- **BE SELECTIVE BUT THOROUGH:** Use 5 to 6 categories, with UP TO 5 concepts per category. Choose the concepts a student must know for an exam - leave out restatements and minor variations of the same idea, but make sure every major topic of the material is represented.
- **ONE SENTENCE PER CONCEPT:** Each concept explanation must be a single short sentence.
- **SHORT OVERVIEW:** The overview must be no more than 3 sentences, written as ONE continuous paragraph on a single line. Do NOT put each sentence on its own line and do NOT insert line breaks anywhere inside the overview.
- **ONLY USE HYPHENS FOR BULLETS:** You MUST use only hyphens (`-`) for ALL bullet points. Do NOT use asterisks (*), bullet symbols (•), or any other characters. EVERY bullet point must start with a hyphen.
- **Bullet Point Format:** Each bullet point must follow this exact format: `- *Concept:* Brief explanation`
- **New Lines:** Ensure every bullet point is on a new line with proper spacing.
- **Readability:** The final output must be scan-friendly and easy to reference quickly.
- **ABSOLUTELY NO MATHEMATICAL NOTATION:** You MUST NOT include ANY mathematical equations, formulas, symbols, or LaTeX notation of any kind. This is CRITICAL - do NOT use:
  * Dollar signs around variables: NO $x$, $\delta$, $P_t$, $\alpha$, etc.
  * Backslash notation: NO \(x\), \[equation\], \delta, \alpha, etc.
  * Mathematical symbols: NO √, ∑, ∫, ≤, ≥, ≠, π, etc.
  * If the material contains LaTeX like $P_t$ or $N_d2$, convert to plain text like "Pt" or "Nd2" (just remove the dollar signs)
  * For Greek letters like $\delta$ or $\alpha$, write out the full word: "delta" or "alpha"
  * Use ONLY plain text to describe all mathematical concepts

---

### REQUIRED OUTPUT STRUCTURE

You MUST use the following structure and formatting precisely.

***OVERVIEW***

Summarise the main subject/topic of the material (for an exam paper or question sheet, the areas its questions cover; for a research article, what it investigates and finds). Write it in a professional, academic tone suitable for quick review before an exam.

***KEY CONCEPTS***

List the most important concepts with brief definitions, grouped into 5-6 logical categories. Each concept must be on its own line. Follow this exact template, continuing the numbering for each category (5 categories minimum, 6 maximum, each with 3-5 concepts):

**1. [First Category Name]**

- *[Concept]:* Brief explanation of the concept.
- *[Concept]:* Brief explanation of the concept.
- *[Concept]:* Brief explanation of the concept.
- *[Concept]:* Brief explanation of the concept.

**2. [Second Category Name]**

- *[Concept]:* Brief explanation of the concept.
- *[Concept]:* Brief explanation of the concept.
- *[Concept]:* Brief explanation of the concept.

**3. [Third Category Name]**

- *[Concept]:* Brief explanation of the concept.
- *[Concept]:* Brief explanation of the concept.
- *[Concept]:* Brief explanation of the concept.

(...continue with categories 4, 5 and, if needed, 6 in the same format...)

End your response with: "Would you like to explore any of these topics in more detail?"

            """

    async def generate_essay_question_async(self):

        if not self.context:
            return "No study material available to create essay question from."

        try:
            messages = self._build_feature_messages(
                "You are an expert at creating analytical essay questions from academic content.",
                self._get_essay_prompt()
            )

            logging.info(f"ASYNC ESSAY: Context length: {len(self._get_truncated_context())} characters")

            # Try the primary model
            try:
                logging.info(f"ASYNC ESSAY: Trying {MODEL_PRIMARY} for essay generation...")
                result = await self._make_async_openai_fallback_call(
                    messages=messages, model=MODEL_PRIMARY, temperature=0.4, max_tokens=15000, timeout=60
                )
                if result and result.strip():
                    logging.info(f"ASYNC ESSAY: {MODEL_PRIMARY} succeeded")
                    return _normalize_study_formatting(_strip_code_fences(result), mode='essay')
                raise ValueError(f"{MODEL_PRIMARY} returned an empty response")
            except Exception as mini_error:
                logging.error(f"ASYNC ESSAY: {MODEL_PRIMARY} failed: {mini_error}")

            # Fallback model
            try:
                logging.info(f"ASYNC ESSAY: Trying {MODEL_FALLBACK} fallback...")
                result = await self._make_async_openai_fallback_call(
                    messages=messages, model=MODEL_FALLBACK, temperature=0.4, max_tokens=15000, timeout=60
                )
                logging.info(f"ASYNC ESSAY: {MODEL_FALLBACK} fallback succeeded")
                return _normalize_study_formatting(_strip_code_fences(result), mode='essay')
            except Exception as nano_error:
                logging.error(f"ASYNC ESSAY: All models failed: {nano_error}")
                return "I'm having trouble generating an essay question right now. The document appears to be loaded successfully, but there may be a temporary issue with the AI service. Please try again in a moment or use the chat to ask specific questions about your document."

        except Exception as e:
            logging.error(f"ASYNC ESSAY: Critical error in async essay generation: {e}")
            return f"Critical error in essay generation: {str(e)}"

    async def generate_essay_question_stream_async(self):
        """Streaming version of generate_essay_question_async. Yields text chunks."""
        if not self.context:
            yield "No study material available to create essay question from."
            return

        try:
            messages = self._build_feature_messages(
                "You are an expert at creating analytical essay questions from academic content.",
                self._get_essay_prompt()
            )

            def factory(model):
                return self._make_async_openai_streaming_call(
                    messages=messages, model=model, temperature=0.4, max_tokens=15000, timeout=60
                )

            async for chunk in _normalize_study_stream(self._stream_with_fallback(
                factory,
                "STREAM ESSAY",
                "I'm having trouble generating an essay question right now. Please try again in a moment."
            ), mode='essay'):
                yield chunk
        except Exception as e:
            logging.error(f"STREAM ESSAY: Critical error: {e}")
            yield f"Critical error in essay generation: {str(e)}"

    def _get_essay_prompt(self):
        """Return the essay question prompt text (single source for streaming and non-streaming)."""
        return r"""

            CRITICAL FORMATTING REQUIREMENTS:
            - Use markdown formatting for emphasis: **bold text**, *italic text*, `code text`
            - **ABSOLUTELY NO MATHEMATICAL NOTATION:** Do NOT use LaTeX formatting, mathematical symbols, or any notation:
              * NO dollar signs: $x$, $\delta$, $P_t$, etc.
              * NO backslash notation: \(x\), \[equation\], etc.
              * NO mathematical symbols: √, ∑, ∫, ≤, ≥, ≠, π, etc.
            - NEVER use HTML tags - only use markdown formatting
            - ONLY USE HYPHENS FOR BULLETS (-) - never use asterisks (*) or dots (•)
            - Each bullet point must be on its own line with consistent hyphen formatting
            - Number sub-questions with DIGITS in the format *1: [question]* - NEVER spell numbers as words (never "One:", "Two:", "Three:")
            - Use ONLY plain English words to describe ALL mathematical concepts
            - Always respond in plain text with markdown formatting only

            TASK:

            **ESSAY QUESTION:**

            Create ONE substantial essay question, with several suggested sub-questions, that:
            - Requires integration of multiple concepts from the study material (for an exam paper or question sheet, the topics its questions test), and
            - Asks for the citation of additional reading of other academic literature, and
            - Asks for commentary on real-world applications
            - Asks for analysis, evaluation, or application (not just description)
            - Is answerable in 500-750 words

---

### REQUIRED OUTPUT STRUCTURE

You MUST use the following structure and formatting precisely.

***ESSAY QUESTION***

**MAIN QUESTION**

[The primary essay prompt — a single, clearly worded question that integrates multiple concepts from the study material and invites critical analysis]

Example format:

*[The question]*

    [2-3 sentences of specific guidance consistent with the structure that will be used for the sub-questions — e.g. "Begin by defining X and Y from the material, then compare how they interact in the context of Z. Use a real-world example such as... to illustrate your argument."]


**SUB-QUESTIONS**

The sub-questions should break down the main question into smaller, manageable parts. For each sub-question, provide:

*The sub-question itself*

    A specific suggestion explaining what the student should do to answer this sub-question well. Reference which concepts from the material to draw on, what kind of analysis is expected, and what evidence or examples to include.

Example format:

*1: [The question]*

    [2-3 sentences of specific guidance — e.g. "Begin by defining X and Y from the material, then compare how they interact in the context of Z. Use a real-world example such as... to illustrate your argument."]

*2: [The question]*

    [2-3 sentences of specific guidance]

*3: [The question]*

    [2-3 sentences of specific guidance]


**ASSESSMENT CRITERIA (QUB Conceptual Equivalents Scale)**

Explain clearly how the essay will be assessed using the QUB Conceptual Equivalents Scale. For each grade band, describe what a student must demonstrate AND give a concrete suggestion for how to achieve that level in THIS specific essay. Do not use bullet points. Provide a paragraph of explanation for each grade band.

**First Class (70-100%):** Exceptional and exemplary work showing a very high level of critical analysis; a very high level of insight in the conclusions drawn; an in-depth knowledge and understanding across a wide range of relevant areas including areas at the forefront of the discipline; very thorough coverage of the topic; and confidence in the appropriate use of learning resources to support arguments made. To achieve this, the student should critically evaluate competing theoretical perspectives, draw on at least 4-5 additional academic references beyond the lecture material, identify limitations or tensions between theories, and demonstrate genuine original insight in their conclusions.

**Upper Second (2:1, 60-69%):** Good performance showing some independence of thought and critical judgement; some ability to analyse concepts and ideas; an understanding of the main issues involved and their relevance; appropriate use of learning resources; and clear understanding of a reasonable range of literature or source materials. To achieve this, the student should go beyond describing concepts to evaluating their strengths and limitations, and reference at least 1-2 sources beyond the lecture material.

**Lower Second (2:2, 50-59%):** Adequate answer showing some knowledge and understanding of the central issues and themes; limited critical analysis and evaluation; limited literature covered; average understanding of materials; and limited independence of thought. The student describes the key concepts correctly but does not evaluate them or connect them to wider literature.

**Third Class (40-49%):** Weak answer showing demonstration of basic knowledge; limited understanding of the topic area; some irrelevance of content; uncritical use of sources; and little indication of independent learning.


**OVERALL ANSWER STRATEGY**

Provide a suggested essay structure with 3-5 concise, actionable tips for how the student should plan and write their answer. For example:
- How to structure the introduction (what to include in the opening paragraph)
- How to organise the body paragraphs around the sub-questions
- How to integrate academic references effectively
- How to write a strong conclusion that demonstrates critical judgement
- What common mistakes to avoid


            RESPONSE FORMAT: Provide the formatted text directly - no JSON, no code blocks."""

    async def explain_key_concepts_stream_async(self):
        """Streaming version of explain_key_concepts_async. Yields text chunks."""
        if not self.context:
            yield "No study material available to explain key concepts from."
            return

        try:
            messages = self._build_feature_messages(
                "You are a patient AI tutor helping students understand the key concepts in their study material (lecture notes, an exam paper, a tutorial/homework question sheet or a research article).",
                self._get_key_concepts_prompt()
            )

            def factory(model):
                return self._make_async_openai_streaming_call(
                    messages=messages, model=model, temperature=0.4, max_tokens=15000, timeout=60
                )

            async for chunk in _normalize_study_stream(self._stream_with_fallback(
                factory,
                "STREAM KEY CONCEPTS",
                "I'm having trouble explaining the key concepts right now. Please try again in a moment."
            )):
                yield chunk
        except Exception as e:
            logging.error(f"STREAM KEY CONCEPTS: Critical error: {e}")
            yield f"Critical error in key concepts explanation: {str(e)}"

    def _get_key_concepts_prompt(self):
        """Return the key concepts prompt text (single source for streaming and non-streaming)."""
        return r"""
Identify and briefly explain exactly 5 key concepts from this study material (for an exam paper or question sheet, the 5 most important concepts its questions test; for a research article, the 5 concepts needed to understand it). Present them in a clear, accessible way that helps students understand complex ideas without being condescending.

CRITICAL FORMATTING REQUIREMENTS:
- **ABSOLUTELY NO MATHEMATICAL NOTATION:** Do NOT use LaTeX formatting, mathematical equations, symbols, or any notation anywhere in your response
  * NO dollar signs around variables: $x$, $\delta$, $P_t$, etc.
  * NO backslash notation: \(x\), \[equation\], etc.
  * NO mathematical symbols: √, ∑, ∫, ≤, ≥, ≠, π, etc.
  * If the material contains LaTeX variables like $P_t$ or $N_d2$, convert to plain text like "Pt" or "Nd2" (just remove dollar signs)
  * For Greek letters like $\delta$ or $\alpha$, write out the full word: "delta" or "alpha"
- Use only plain text with markdown formatting (bold, italic, bullet points)
- NEVER use hash heading syntax (#, ##, ###) anywhere in the response
- Do NOT wrap any part of your response in code fences or backticks - output the text directly
- ONLY USE HYPHENS FOR BULLETS (-) - never use asterisks (*) or dots (•)
- Each bullet point must be on its own line with consistent hyphen formatting
- Describe ALL mathematical concepts using ONLY plain English words
- CONCEPT HEADINGS: Each concept heading must be bold, on its own line (NOT a bullet point), numbered with a DIGIT and a period, exactly like: **1. Concept Name**
- The no-mathematical-notation rule does NOT apply to these heading numbers: you MUST use the digits 1. 2. 3. 4. 5. - NEVER spell them as words (never "One:", "Two:", "Three:")

Use the following structure. Each bullet point MUST be on its own line.

***KEY CONCEPTS EXPLAINED:***

**1. [First Concept]**

  - *What it is:* Clear, straightforward definition in plain language

  - *How it works:* Brief explanation of the concept's mechanism or process

  - *Why it matters:* Practical significance and relevance

  - *Real-world example:* Concrete example that illustrates the concept.


**2. [Second Concept]**

  - *What it is:* Clear, straightforward definition in plain language

  - *How it works:* Brief explanation of the concept's mechanism or process

  - *Why it matters:* Practical significance and relevance

  - *Real-world example:* Concrete example that illustrates the concept.


**3. [Third Concept]**

  - *What it is:* Clear, straightforward definition in plain language

  - *How it works:* Brief explanation of the concept's mechanism or process

  - *Why it matters:* Practical significance and relevance

  - *Real-world example:* Concrete example that illustrates the concept.

(Continue the same pattern for concepts 4 and 5, so that exactly 5 key concepts are covered.)

End the overall response with: "Would you like to explore any of these topics in more detail?"
"""

    async def explain_key_concepts_async(self):

        if not self.context:
            return "No study material available to explain key concepts from."

        try:
            messages = self._build_feature_messages(
                "You are a patient AI tutor helping students understand the key concepts in their study material (lecture notes, an exam paper, a tutorial/homework question sheet or a research article).",
                self._get_key_concepts_prompt()
            )

            # Try the primary model
            try:
                logging.info(f"ASYNC KEY CONCEPTS: Trying {MODEL_PRIMARY} for key concepts explanation...")
                logging.info(f"ASYNC KEY CONCEPTS: Context length: {len(self._get_truncated_context())} characters")

                result = await self._make_async_openai_fallback_call(
                    messages=messages,
                    model=MODEL_PRIMARY,
                    temperature=0.4,
                    max_tokens=15000,
                    timeout=60
                )

                logging.info(f"ASYNC KEY CONCEPTS: {MODEL_PRIMARY} succeeded")
                return _normalize_study_formatting(_strip_code_fences(result))

            except Exception as openai_error:
                logging.error(f"ASYNC KEY CONCEPTS: {MODEL_PRIMARY} failed: {openai_error}")
                # Fallback model
                try:
                    logging.info(f"ASYNC KEY CONCEPTS: Trying {MODEL_FALLBACK} fallback...")
                    fallback_result = await self._make_async_openai_fallback_call(
                        messages=messages,
                        model=MODEL_FALLBACK,
                        temperature=0.4,
                        max_tokens=15000,
                        timeout=60
                    )

                    logging.info(f"ASYNC KEY CONCEPTS: {MODEL_FALLBACK} fallback succeeded")
                    return _normalize_study_formatting(_strip_code_fences(fallback_result))

                except Exception as nano_error:
                    logging.error(f"ASYNC KEY CONCEPTS: Both {MODEL_PRIMARY} and {MODEL_FALLBACK} failed: {nano_error}")
                    return "I'm having trouble explaining the key concepts right now. The document appears to be loaded successfully, but there may be a temporary issue with the AI service. Please try again in a moment or use the chat to ask specific questions about your document."

        except Exception as e:
            logging.error(f"ASYNC KEY CONCEPTS: Critical error in async key concepts explanation: {e}")
            return f"Critical error in key concepts explanation: {str(e)}"

    @staticmethod
    def _parse_and_validate_quiz(content):
        """Parse a quiz JSON response and validate/repair its questions.

        Shared by the primary and fallback quiz paths. One malformed question
        is skipped rather than aborting the whole batch.
        """
        cleaned_content = content.strip()

        # Remove common markdown code block patterns
        if cleaned_content.startswith('```json'):
            cleaned_content = cleaned_content[7:]
        if cleaned_content.startswith('```'):
            cleaned_content = cleaned_content[3:]
        if cleaned_content.endswith('```'):
            cleaned_content = cleaned_content[:-3]

        # Remove any leading text before the first {
        first_brace = cleaned_content.find('{')
        if first_brace > 0:
            cleaned_content = cleaned_content[first_brace:]

        # Remove any trailing text after the last }
        last_brace = cleaned_content.rfind('}')
        if last_brace != -1 and last_brace < len(cleaned_content) - 1:
            cleaned_content = cleaned_content[:last_brace + 1]

        parsed_result = json.loads(cleaned_content)
        questions = parsed_result.get("questions", [])[:15]  # Limit to 15 questions

        def _strip_option_prefixes(text):
            for prefix in ('Option A:', 'Option B:', 'Option C:', 'Option D:'):
                text = text.replace(prefix, '')
            return text.strip()

        valid_questions = []
        for i, q in enumerate(questions):
            try:
                # Check required fields
                if not isinstance(q, dict) or not all(key in q for key in ("question", "options", "correct_answer", "explanation")):
                    logging.warning(f"Quiz question {i+1} missing required fields")
                    continue

                # Validate options structure
                if not isinstance(q["options"], list) or len(q["options"]) != 4:
                    logging.warning(f"Quiz question {i+1} options invalid")
                    continue
                if not all(isinstance(opt, str) for opt in q["options"]):
                    logging.warning(f"Quiz question {i+1} has non-string options")
                    continue

                # Coerce numeric correct_answer values to strings; skip others
                correct_answer = q["correct_answer"]
                if not isinstance(correct_answer, str):
                    if isinstance(correct_answer, (int, float)) and not isinstance(correct_answer, bool):
                        correct_answer = str(correct_answer)
                        q["correct_answer"] = correct_answer
                    else:
                        logging.warning(f"Quiz question {i+1} correct_answer is not a string")
                        continue

                correct_answer = correct_answer.strip()
                options = [opt.strip() for opt in q["options"]]

                # Validate correct_answer matches one of the options
                if correct_answer not in options:
                    # Try to reconcile "Option A:"-style prefixes
                    matched = False
                    target = _strip_option_prefixes(correct_answer)
                    for option in options:
                        if _strip_option_prefixes(option) == target:
                            q["correct_answer"] = option  # Fix the correct answer
                            matched = True
                            break
                    if not matched:
                        logging.warning(f"Quiz question {i+1} correct_answer mismatch")
                        continue

                # Validate no empty fields
                if not isinstance(q["question"], str) or not isinstance(q["explanation"], str) or not q["question"].strip() or not q["explanation"].strip():
                    logging.warning(f"Quiz question {i+1} has empty fields")
                    continue

                # Shuffle the (stripped) options so the model's position bias
                # (correct answer listed first) never reaches students
                if correct_answer in options:
                    q["correct_answer"] = correct_answer
                random.shuffle(options)
                q["options"] = options

                valid_questions.append(q)
            except Exception as question_error:
                logging.warning(f"Quiz question {i+1} skipped due to validation error: {question_error}")
                continue

        return valid_questions

    async def generate_retrieval_quiz_async(self):

        if not self.context:
            return []

        try:
            context_truncated = self._get_truncated_context()

            prompt = rf"""Based on this study material, create exactly 15 simple multiple choice questions about the key concepts. Do NOT include any mathematical equations or formulas.

{self._material_guidance()}{context_truncated}

CRITICAL INSTRUCTIONS:
1. Your response must start immediately with {{ and end with }} - NO other characters
2. Do NOT include any text, explanations, or formatting before or after the JSON
3. Do NOT include markdown code blocks, backticks, or any other formatting
4. Do NOT include any HTML tags or special characters outside the JSON
5. Use this EXACT JSON structure and format:

{{
    "questions": [
        {{
            "question": "What is the main topic of this material?",
            "options": ["First answer choice", "Second answer choice", "Third answer choice", "Fourth answer choice"],
            "correct_answer": "Second answer choice",
            "explanation": "Brief explanation without saying 'Correct!' at the beginning"
        }}
    ]
}}

ANSWER FORMAT REQUIREMENTS:
- The "correct_answer" field must EXACTLY match one of the options in the "options" array
- Options should contain the actual answer text without any prefixes like "Option A:" or "Option B:"
- The correct_answer must be the EXACT text from the options array
- Example: If options are ["Risk increases", "Risk decreases", "No change", "Unknown"], then correct_answer must be one of these exact strings like "Risk increases"
- All four options MUST be of similar length and detail (within a few words of each other). Do NOT make the correct answer longer, more precise, or more carefully worded than the other options - students spot this pattern. Write every incorrect option with the same level of detail and plausibility as the correct one
- Vary the position of the correct answer across questions - it must NOT usually be the first option

Make questions simple and focused on basic concepts. Keep explanations short and informative.

CRITICAL FORMATTING RULES:
- Do NOT start explanations with ANY confirmation words like 'Correct!', 'Right!', 'Yes!', 'That's correct!', 'Exactly!', or any similar phrases. Start explanations directly with the educational content.
- **ABSOLUTELY NO MATHEMATICAL NOTATION:** Do NOT use LaTeX formatting, mathematical symbols, or any notation in questions, options, or explanations:
  * NO dollar signs: $x$, $\delta$, $P_t$, etc.
  * NO backslash notation: \(x\), \[equation\], etc.
  * NO mathematical symbols: √, ∑, ∫, ≤, ≥, ≠, π, etc.
  * If the material contains LaTeX variables like $P_t$ or $N_d2$, convert to plain text like "Pt" or "Nd2" (just remove dollar signs)
  * For Greek letters like $\delta$ or $\alpha$, write out the full word: "delta" or "alpha"
- Use ONLY plain English words to describe all mathematical concepts
- Use only plain text - no mathematical notation or formulas

RESPONSE FORMAT: Start your response with {{ immediately - no whitespace, no text, no code blocks."""

            messages = [{"role": "user", "content": prompt}]

            # Try the primary model
            try:
                logging.info(f"ASYNC QUIZ: Trying {MODEL_PRIMARY} for retrieval quiz generation...")
                logging.info(f"ASYNC QUIZ: Context length: {len(context_truncated)} characters")

                result = await self._make_async_openai_fallback_call(
                    messages=messages,
                    model=MODEL_PRIMARY,
                    response_format={"type": "json_object"},
                    temperature=0.3,
                    max_tokens=15000,
                    timeout=60
                )

                logging.info(f"ASYNC QUIZ: {MODEL_PRIMARY} succeeded")
                valid_questions = self._parse_and_validate_quiz(result)
                logging.info(f"ASYNC QUIZ: Generated {len(valid_questions)} valid questions")
                return valid_questions

            except Exception as primary_error:
                logging.error(f"ASYNC QUIZ: {MODEL_PRIMARY} failed: {primary_error}")
                # Fallback model
                try:
                    logging.info(f"ASYNC QUIZ: Trying {MODEL_FALLBACK} fallback...")
                    fallback_result = await self._make_async_openai_fallback_call(
                        messages=messages,
                        model=MODEL_FALLBACK,
                        response_format={"type": "json_object"},
                        temperature=0.3,
                        max_tokens=15000,
                        timeout=60
                    )

                    logging.info(f"ASYNC QUIZ: {MODEL_FALLBACK} fallback succeeded")
                    valid_questions = self._parse_and_validate_quiz(fallback_result)
                    logging.info(f"ASYNC QUIZ: Fallback generated {len(valid_questions)} valid questions")
                    return valid_questions

                except Exception as fallback_error:
                    logging.error(f"ASYNC QUIZ: Both async methods failed: {fallback_error}")
                    return []

        except Exception as e:
            logging.error(f"ASYNC QUIZ: Critical error in async quiz generation: {e}")
            return []

    # Legacy sync method removed - all quiz operations now use async polling

    async def detect_document_type_async(self):
        """Detect whether the uploaded document is an exam paper, a tutorial/exercise/homework
        question sheet, a research article, or lecture notes. Returns 'exam_paper', 'exercise_set',
        'research_article' or 'lecture_notes'."""
        if not self.context:
            return "lecture_notes"

        try:
            sample = self.context[:8000]
            prompt = f"""Classify this document as EXACTLY ONE of "exam_paper", "exercise_set", "research_article" or "lecture_notes".

exam_paper: a formal examination or class test. It contains numbered questions asking students to
calculate, solve, evaluate or discuss specific problems, and typically has marks allocated
(e.g. [10 marks]), a time limit, instructions like "Answer ALL questions", and formal exam rubric.

exercise_set: a tutorial sheet, exercise set, problem set, seminar/workshop questions, worksheet or
homework/assignment. It is ALSO mainly a list of questions or problems for students to attempt,
but is not a formal timed exam - it may be titled "Tutorial 3", "Exercises", "Problem Set 2",
"Homework", "Seminar questions", etc., and may or may not include solutions.

research_article: an academic journal article, working paper or similar research paper. It typically
has authors and affiliations, an abstract, keywords or JEL codes, an introduction and literature
review, data and methodology sections, empirical or theoretical results, a conclusion and a
reference list.

lecture_notes: slides or notes containing explanations, theory, derivations, definitions and
worked examples used for teaching - NOT primarily a set of questions to be answered and NOT a
research paper.

DOCUMENT EXCERPT:
{sample}

Reply with ONLY one of these four strings, nothing else:
exam_paper
exercise_set
research_article
lecture_notes"""

            messages = [{"role": "user", "content": prompt}]
            for model in (MODEL_PRIMARY, MODEL_FALLBACK):
                try:
                    content = await self._make_async_openai_fallback_call(
                        messages, model=model, max_tokens=1000, timeout=30
                    )
                    result = content.strip().lower().replace('"', '').replace("'", "")
                    if "research" in result or "article" in result:
                        doc_type = "research_article"
                    elif "exercise" in result or "tutorial" in result or "homework" in result:
                        doc_type = "exercise_set"
                    elif "exam" in result:
                        doc_type = "exam_paper"
                    else:
                        doc_type = "lecture_notes"
                    logging.info(f"Document classified as {doc_type} by {model}")
                    return doc_type
                except Exception as e:
                    logging.error(f"Document classification failed with {model}: {e}")

            return "lecture_notes"
        except Exception as e:
            logging.error(f"detect_document_type_async failed: {e}")
            return "lecture_notes"

    async def extract_exam_questions_async(self):
        """Extract individual calculation questions from an exam paper or exercise/homework sheet, preserving exact wording and numbers."""
        if not self.context:
            return []

        try:
            context_truncated = self._get_truncated_context()

            prompt = f"""You are analysing a {self._material_noun()} to extract each individual calculation question.

DOCUMENT ({self._material_label()}):
{context_truncated}

TASK: Extract every calculation question from this document as a list of self-contained question objects.

Rules:
- Include EVERY question and sub-question that requires a numerical calculation (e.g. 1a, 1b, 2a, 2b, 2c etc.).
- Preserve the EXACT wording, numbers, and data from the document — do not alter, summarise, or rephrase.
- Each entry must be completely self-contained: include any shared data, tables, or context from the parent question that the sub-question depends on so it can be understood in isolation.
- Preserve the original question ordering.
- Exclude purely discursive questions that ask students to "explain", "discuss", or "describe" without any calculation.
- Include the mark allocation if stated (e.g. "[10 marks]").
- Return ONLY a valid JSON array of objects with the keys "id" and "question", no surrounding text.

Example output format:
[
  {{"id": "1a", "question": "A bond with a face value of £1,000 pays a coupon of 5% per annum... Calculate the present value of the bond. [10 marks]"}},
  {{"id": "1b", "question": "Using the same bond from Question 1a (face value £1,000, coupon 5%)... Calculate the yield to maturity if the bond is trading at £950. [8 marks]"}}
]"""

            messages = [{"role": "user", "content": prompt}]

            for model in (MODEL_PRIMARY, MODEL_FALLBACK):
                try:
                    content = await self._make_async_openai_fallback_call(
                        messages, model=model, max_tokens=8000, timeout=90
                    )
                    content = _strip_code_fences(content.strip())
                    questions = json.loads(content)
                    if isinstance(questions, list) and len(questions) > 0:
                        logging.info(f"Extracted {len(questions)} exam questions using {model}")
                        return questions
                except Exception as e:
                    logging.error(f"Exam question extraction failed with {model}: {e}")

            return []

        except Exception as e:
            logging.error(f"extract_exam_questions_async failed: {e}")
            return []

    async def extract_equation_list_async(self):
        """Extract all calculable equations from the lecture notes in order, returning a list of LaTeX strings."""
        if not self.context:
            return []

        try:
            context_truncated = self._get_truncated_context()

            prompt = rf"""You are analysing lecture notes to identify the key mathematical equations a student should practise.

LECTURE NOTES:
{context_truncated}

TASK: List the MAIN equations and formulas from the notes above, in the EXACT ORDER they appear from top to bottom.

Rules:
- Focus on the principal, named equations that are central to the topic — the ones a student would be expected to know and apply in an exam.
- Omit minor rearrangements, trivial sub-steps, or worked-example intermediate lines that are just applications of a main equation already listed.
- Only include equations that contain computable quantities (i.e. a student could substitute in numbers and calculate an answer).
- Exclude purely conceptual definitions that have no numerical calculation involved.
- If the same equation appears in multiple forms, include only the most general or most useful form once.
- Preserve the original order — do not reorder or group them.
- Express each equation in standard LaTeX notation.
- Return ONLY a valid JSON array of strings, with no surrounding text, no markdown, no code fences.

Example output format:
["\\hat{{\\mu}}_{{12}} = \\frac{{n_1 \\hat{{\\mu}}_1 + n_2 \\hat{{\\mu}}_2}}{{n_1 + n_2}}", "\\sigma^2 = \\frac{{\\sum(x_i - \\bar{{x}})^2}}{{n-1}}"]"""

            messages = [{"role": "user", "content": prompt}]

            for model in (MODEL_PRIMARY, MODEL_FALLBACK):
                try:
                    content = await self._make_async_openai_fallback_call(
                        messages, model=model, max_tokens=8000, timeout=60
                    )
                    content = _strip_code_fences(content.strip())
                    equations = json.loads(content)
                    if isinstance(equations, list) and len(equations) > 0:
                        logging.info(f"Extracted {len(equations)} equations using {model}")
                        return equations
                except Exception as e:
                    logging.error(f"Equation extraction failed with {model}: {e}")

            return []

        except Exception as e:
            logging.error(f"extract_equation_list_async failed: {e}")
            return []

    def _get_calculation_question_prompt(self, context_truncated, specific_equation=None, used_questions=None):
        """Return the calculation question prompt (single source for streaming and non-streaming)."""
        if specific_equation:
            equation_instruction = f"""
EQUATION TO USE:
You MUST base this question on the following specific equation from the lecture notes:

    {specific_equation}

Do not choose a different equation — this is the equation for this question."""
        else:
            used_questions_context = ""
            if used_questions and len(used_questions) > 0:
                used_questions_context = f"""
PREVIOUSLY USED QUESTIONS (DO NOT REPEAT):
{chr(10).join([f"{i+1}. {q[:150]}..." for i, q in enumerate(used_questions)])}
"""
            equation_instruction = f"""
{used_questions_context}
Choose ONE equation from the lecture notes that has not been used before."""

        return rf"""Based on the following lecture notes, generate ONE calculation question that follows this specific 6-part layout pattern:

            LECTURE NOTES:
            {context_truncated}

            {equation_instruction}

            REQUIRED LAYOUT PATTERN:
            1) Display the equation you are using in LaTeX formatting.
            2) Explain the variable definitions clearly, listing EACH variable on its own line as a hyphen bullet
            3) Provide an explanation of what the equation means and its purpose
            4) Show a worked example using specific input values with step-by-step LaTeX calculations
            5) Set a challenge for the user using different input values
            6) Ask the user to input their answer in the chat

            CONTENT REQUIREMENTS:
            - Use ONLY the equation specified above — do not substitute a different one
            - For worked examples, you may use values from the lecture notes if available
            - For challenge problems, you MUST create NEW and DIFFERENT numerical values
            - Show complete step-by-step calculations in LaTeX format
            - End by asking the user to type their numerical answer in the chat

            FORMATTING REQUIREMENTS:
            - Use standard LaTeX: \[ equation \] for display math, \( variable \) for inline math
- CRITICAL - LONG EQUATIONS: rendered math cannot wrap, so a long equation must be SPLIT ACROSS LINES by you. If an equation has more than about 5 terms or would be wider than a phone screen, write it as a \begin{{align*}} block broken at = or + or - signs, with each continuation line starting with the operator after the alignment marker (e.g. a first line ending in the left-hand side and = , then continuation lines like &\quad + \text{{next terms}}), indented so it clearly reads as ONE equation continuing over several lines. NEVER emit a single line of math wider than a phone screen
            - Use standard math operators: \times, \div, \cdot, \frac{{numerator}}{{denominator}}
            - For subscripts: Always use underscore with braces \mu_{{12}} (proper braces required)
            - For superscripts: Always use caret with braces \sigma^{{2}} (proper braces required)
            - For combined: \hat{{\mu}}_{{12}} or \sigma_{{1}}^{{2}} (always use proper braces)
            - CRITICAL: Every subscript and superscript MUST have proper braces like _{{value}} and ^{{value}}
            - CRITICAL: Inside math, ALWAYS escape percent signs as \% (write 5\%, never 5%) and ampersands as \& — a bare % or & breaks the rendering
- CRITICAL: In ordinary sentences OUTSIDE math delimiters, write numbers and percentages as plain text (e.g. "a return of 5%", "grows by 12%") — do NOT wrap plain numbers or percentages in \( \); reserve inline math for variables and symbols only
- CRITICAL: Currency — use the real symbols with amounts (£1,000, $500, €250), NOT the words pounds/dollars/euros. £ and € may be written directly inside math; the dollar sign inside math MUST be escaped as \$ (a raw $ inside math breaks the rendering)
            - For the worked examples, you MUST use \begin{{align*}} with proper alignment for each step so that the calculations are clear and easy to follow.
            - For matrices: \begin{{bmatrix}} a & b \\ c & d \end{{bmatrix}}.
            - For line spacing in align* environments use \\[6pt] between lines.
            - For worked examples: Use **Step N:** Description followed by calculation
            - Use **bold** for section headers and step descriptions




            EXAMPLE FORMAT:

            ***CALCULATION QUESTION***

            **EQUATION**
            One of the equations used in this topic is:

            \[ LaTeX equation here \]
            The variables in this equation are:
            - \(x\): Variable description
            - \(y\): Another variable description
            - \(z\): Another variable description


            **EXPLANATION**
            Explanation of the equation's purpose and application.


            **WORKED EXAMPLE**
            Given values from lecture notes: \(x = 10\); \(y = 5\); and \(z = 2\)

            **Step 1:** Calculate the sum
            \begin{{align*}}
            x + y + z &= 10 + 5 + 2\\[6pt]
            &= 17
            \end{{align*}}

            **Step 2:** Multiply by 2
            \begin{{align*}}
            \text{{result}} \times 2 &= 17 \times 2 \\[6pt]
            &= 34
            \end{{align*}}

            **Final Result:**
            \begin{{align*}}
            \text{{Answer}} &= 34
            \end{{align*}}

            **CHALLENGE**
            Calculate the result when: \(x = 12\); \(y = 8\); and \(z = 10\)

            Please type your numerical answer in the chat below.

            RESPONSE FORMAT: Provide the formatted text directly - no JSON, no code blocks, just the formatted calculation question."""

    async def generate_calculation_question_async(self, used_questions=None, specific_equation=None, exam_question=None):

        if not self.context:
            return "No document context available. Please upload a document first."

        try:
            context_truncated = self._get_truncated_context()

            # --- EXAM PAPER MODE ---
            if exam_question:
                return await self._generate_exam_worked_example(context_truncated, exam_question)

            # --- LECTURE NOTES MODE ---
            prompt = self._get_calculation_question_prompt(
                context_truncated, specific_equation=specific_equation, used_questions=used_questions
            )

            messages = [{"role": "user", "content": prompt}]

            try:
                logging.debug(f"ASYNC DEBUG: Trying async {MODEL_PRIMARY} for calculation question generation...")
                result = await self._make_async_openai_fallback_call(
                    messages=messages,
                    model=MODEL_PRIMARY,
                    max_tokens=8000,
                    timeout=120,
                    reasoning_effort="medium"
                )

                # Log raw API response
                logging.debug(f"RAW API RESPONSE:\n{result}")

                # No LaTeX formatting - let MathJax handle delimiters directly
                logging.info(f"Async calculation question generated successfully using {MODEL_PRIMARY}")
                return result

            except Exception as e:
                logging.error(f"Async generation failed after all attempts: {e}")
                # Instead of falling back to sync (which blocks the event loop), return a clean error message
                return "I'm sorry, the AI service is taking too long to generate a calculation question right now. Please try again in a moment."

        except Exception as e:
            logging.error(f"Async calculation question generation failed: {e}")
            return "Sorry, I couldn't generate calculation questions at this time. Please try again later."

    async def generate_calculation_question_stream_async(self, specific_equation=None, used_questions=None):
        """Streaming version of generate_calculation_question_async for lecture notes. Yields text chunks."""

        if not self.context:
            yield "No document context available. Please upload a document first."
            return

        prompt = self._get_calculation_question_prompt(
            self._get_truncated_context(), specific_equation=specific_equation, used_questions=used_questions
        )

        messages = [{"role": "user", "content": prompt}]

        emitted = False
        try:
            logging.info(f"CALC_QUESTION_STREAM: Streaming with {MODEL_PRIMARY} + reasoning_effort=medium...")
            async for chunk in self._make_async_openai_streaming_call(
                messages=messages,
                model=MODEL_PRIMARY,
                max_tokens=8000,
                timeout=120,
                reasoning_effort="medium"
            ):
                emitted = True
                yield chunk
            logging.info("CALC_QUESTION_STREAM: Streaming completed")
        except Exception as e:
            logging.error(f"CALC_QUESTION_STREAM: Streaming failed: {e}")
            if emitted:
                yield "\n\n[Connection interrupted - please ask me to continue]"
            else:
                yield "I'm sorry, the AI service is taking too long to generate a calculation question right now. Please try again in a moment."

    def _get_exam_worked_example_prompt(self, context_truncated, exam_question):
        """Return the exam worked example prompt (single source for streaming and non-streaming)."""
        q_id = exam_question.get("id", "?")
        q_text = exam_question.get("question", "")

        return rf"""You are helping a student work through their {self._material_noun()}. Below is the full document for context, followed by ONE specific question from it.

DOCUMENT ({self._material_label()}, for reference/context):
{context_truncated}

SPECIFIC QUESTION TO SOLVE (Question {q_id}):
{q_text}

YOUR TASK:
1. Produce a complete worked solution for THIS EXACT question using the EXACT numbers and data given.
2. If the document already includes a solution or answer for this question, cross-check YOUR calculated answer against the PROVIDED solution. If they differ, carefully re-examine both approaches, identify where any error lies (yours or the paper's), and explain the discrepancy to the student.
3. Then set the student a new challenge using different numbers.

REQUIRED LAYOUT:

***EXAM QUESTION {q_id}***

**QUESTION**
Reproduce the exact question text here so the student can read it.

**EQUATION**
Display the key equation(s) needed to solve this question in LaTeX.

The variables in this equation are:
- \(x\): description
- \(y\): description
(one hyphen bullet per variable, each on its own line, directly below the equation with no blank lines in between)

**EXPLANATION**
Briefly explain what the equation does and why it is used here.

**WORKED SOLUTION**
Solve the question step-by-step using the EXACT numbers from the exam question.

**Step 1:** Description
\begin{{align*}}
calculation &= ... \\[6pt]
&= ...
\end{{align*}}

(Continue with as many steps as needed to reach the final answer.)

**Final Result:**
\begin{{align*}}
\text{{Answer}} &= ...
\end{{align*}}

**SOLUTION VERIFICATION** (include this section ONLY if the document provides a solution or answer)
Compare your calculated answer with the solution provided in the document. If they match, confirm this. If they differ, explain clearly where the discrepancy is, which approach contains the error, and why.

**CHALLENGE**
Now try a similar problem with DIFFERENT numerical values (you invent new, realistic values).
State the new values clearly, then ask the student to calculate the answer.

Please type your numerical answer in the chat below.

FORMATTING REQUIREMENTS:
- Use standard LaTeX: \[ equation \] for display math, \( variable \) for inline math
- CRITICAL - LONG EQUATIONS: rendered math cannot wrap, so a long equation must be SPLIT ACROSS LINES by you. If an equation has more than about 5 terms or would be wider than a phone screen, write it as a \begin{{align*}} block broken at = or + or - signs, with each continuation line starting with the operator after the alignment marker (e.g. a first line ending in the left-hand side and = , then continuation lines like &\quad + \text{{next terms}}), indented so it clearly reads as ONE equation continuing over several lines. NEVER emit a single line of math wider than a phone screen
- Use \begin{{align*}} with \\[6pt] line spacing for multi-step calculations
- Use **bold** for section headers and step descriptions
- For matrices: \begin{{bmatrix}} a & b \\ c & d \end{{bmatrix}}
- CRITICAL: Every subscript and superscript MUST have proper braces like _{{value}} and ^{{value}}
- CRITICAL: Inside math, ALWAYS escape percent signs as \% (write 5\%, never 5%) and ampersands as \& — a bare % or & breaks the rendering
- CRITICAL: In ordinary sentences OUTSIDE math delimiters, write numbers and percentages as plain text (e.g. "a return of 5%", "grows by 12%") — do NOT wrap plain numbers or percentages in \( \); reserve inline math for variables and symbols only
- CRITICAL: Currency — use the real symbols with amounts (£1,000, $500, €250), NOT the words pounds/dollars/euros. £ and € may be written directly inside math; the dollar sign inside math MUST be escaped as \$ (a raw $ inside math breaks the rendering)
- Provide the formatted text directly — no JSON, no code blocks."""

    async def _generate_exam_worked_example(self, context_truncated, exam_question):
        """Generate a worked example for an exam paper question using its exact numbers, then set a challenge with different numbers."""
        q_id = exam_question.get("id", "?")
        prompt = self._get_exam_worked_example_prompt(context_truncated, exam_question)
        messages = [{"role": "user", "content": prompt}]

        try:
            logging.info(f"Generating exam worked example for question {q_id}...")
            result = await self._make_async_openai_fallback_call(
                messages=messages,
                model=MODEL_PRIMARY,
                max_tokens=8000,
                timeout=120,
                reasoning_effort="medium"
            )
            logging.info(f"Exam worked example generated successfully for question {q_id}")
            return result
        except Exception as e:
            logging.error(f"Exam worked example generation failed: {e}")
            return "I'm sorry, the AI service is taking too long to generate a worked example right now. Please try again in a moment."

    async def _generate_exam_worked_example_stream(self, context_truncated, exam_question):
        """Streaming version of _generate_exam_worked_example. Yields text chunks."""
        q_id = exam_question.get("id", "?")
        prompt = self._get_exam_worked_example_prompt(context_truncated, exam_question)
        messages = [{"role": "user", "content": prompt}]

        emitted = False
        try:
            logging.info(f"Streaming exam worked example for question {q_id}...")
            async for chunk in self._make_async_openai_streaming_call(
                messages=messages,
                model=MODEL_PRIMARY,
                max_tokens=8000,
                timeout=120,
                reasoning_effort="medium"
            ):
                emitted = True
                yield chunk
            logging.info(f"Exam worked example streaming completed for question {q_id}")
        except Exception as e:
            logging.error(f"Exam worked example streaming failed: {e}")
            if emitted:
                yield "\n\n[Connection interrupted - please ask me to continue]"
            else:
                yield "I'm sorry, the AI service is taking too long to generate a worked example right now. Please try again in a moment."

    def _get_answer_check_prompt(self, truncated_context, challenge_question, user_answer):
        """Return the calculation answer-check prompt (single source for streaming and non-streaming)."""
        return rf"""You are evaluating a student's answer to a calculation question.

LECTURE NOTES CONTEXT:
{truncated_context}

ORIGINAL CHALLENGE QUESTION:
{challenge_question}

STUDENT'S ANSWER: {user_answer}

EVALUATION TASKS:
1. Calculate the correct answer using the provided challenge values
2. Provide the correct step-by-step solution
3. Determine if the student's answer is correct (accept reasonable rounding and alternative valid approaches)
4. Give appropriate feedback
5. Ask if they want another calculation question

NOTE: The student may provide a numerical answer, a formula, a written explanation of their working, or a combination. Evaluate whatever form of answer they have given.

RESPONSE FORMAT:
**Step-by-Step Solution:**
**Step 1:** [Description of first step]
\begin{{align*}}
[equation 1] &= [step 1] \\
&= [step 2] \\
&= [final result]
\end{{align*}}
**Step 2:** [Description of second step]
\begin{{align*}}
[equation 2] &= [step 1] \\
&= [step 2] \\
&= [final result]
\end{{align*}}
[Continue for all steps...]

**Final Answer:** [Correct numerical answer]

**Feedback:** [Comment on the student's answer]

**To move on to the next set of equations from the lecture notes, just click the Next Question button.**

CRITICAL FORMATTING REQUIREMENTS:
- ALWAYS use \begin{{align*}} environment for ALL step-by-step calculations
- Use standard LaTeX: \( variable \) for inline math in text descriptions
- Use standard math operators: \times, \div, \cdot, \frac{{numerator}}{{denominator}}
- For subscripts: Always use underscore with braces \mu_{{12}} (proper braces required)
- For superscripts: Always use caret with braces \sigma^{{2}} (proper braces required)
- For combined: \hat{{\mu}}_{{12}} or \sigma_{{1}}^{{2}} (always use proper braces)
- CRITICAL: Every subscript and superscript MUST have proper braces like _{{value}} and ^{{value}}
- CRITICAL: Inside math, ALWAYS escape percent signs as \% (write 5\%, never 5%) and ampersands as \& — a bare % or & breaks the rendering
- CRITICAL: In ordinary sentences OUTSIDE math delimiters, write numbers and percentages as plain text (e.g. "a return of 5%", "grows by 12%") — do NOT wrap plain numbers or percentages in \( \); reserve inline math for variables and symbols only
- CRITICAL: Currency — use the real symbols with amounts (£1,000, $500, €250), NOT the words pounds/dollars/euros. £ and € may be written directly inside math; the dollar sign inside math MUST be escaped as \$ (a raw $ inside math breaks the rendering)
- For matrices: \begin{{bmatrix}} a & b \\ c & d \end{{bmatrix}}
- For line spacing in align* environments use \\[6pt] between lines.
- Use **bold** for section headers and step descriptions
- Each step must be in its own align* environment for proper formatting

IMPORTANT MULTI-LINE CALCULATION FORMATTING:
INCORRECT:
\[
K\,e^{{-rT}}
= 52 \times e^{{-0.05 \times 1}}
= 52 \times e^{{-0.05}}
\approx 52 \times 0.951229
\approx 49.4629
\]

CORRECT:
\begin{{align*}}
    K e^{{-rT}} &= 52 \times e^{{-0.05 \times 1}} \\
              &= 52 \times e^{{-0.05}} \\
              &\approx 52 \times 0.951229 \\
              &\approx 49.4629
\end{{align*}}

"""

    async def check_calculation_answer_async(self, challenge_question, user_answer):

        if not self.context:
            return "No document context available."

        prompt = self._get_answer_check_prompt(self._get_truncated_context(ANSWER_CHECK_CONTEXT_CHARS), challenge_question, user_answer)
        messages = [{"role": "user", "content": prompt}]

        try:
            logging.info(f"CHECK_CALC_ANSWER: Trying async {MODEL_PRIMARY} for calculation answer evaluation...")
            response = await self._make_async_openai_fallback_call(
                messages=messages,
                model=MODEL_PRIMARY,
                max_tokens=8000,
                timeout=120,
                reasoning_effort="medium"
            )

            logging.debug("CHECK_CALC_ANSWER: RAW API RESPONSE:")
            logging.debug(response)

            # No LaTeX formatting - let MathJax handle delimiters directly
            logging.info(f"CHECK_CALC_ANSWER: {MODEL_PRIMARY} success - returning raw response")
            return response

        except Exception as primary_error:
            logging.error(f"{MODEL_PRIMARY} failed for calculation answer check: {primary_error}")
            # Fallback model
            try:
                logging.info(f"CHECK_CALC_ANSWER: Falling back to {MODEL_FALLBACK}...")
                fallback_response = await self._make_async_openai_fallback_call(
                    messages=messages,
                    model=MODEL_FALLBACK,
                    max_tokens=8000,
                    timeout=120,
                    reasoning_effort="medium"
                )

                logging.debug(f"CHECK_CALC_ANSWER: {MODEL_FALLBACK} fallback success")
                return fallback_response

            except Exception as fallback_error:
                logging.error(f"Both {MODEL_PRIMARY} and {MODEL_FALLBACK} failed for calculation answer check: {primary_error} | {fallback_error}")
                return f"**Feedback:** I received your answer: {user_answer}. However, I'm having trouble processing calculation evaluations right now. Please try again in a moment, or click the Calculation questions button to get a new question."

    async def check_calculation_answer_stream_async(self, challenge_question, user_answer):
        """Streaming version of check_calculation_answer_async. Yields text chunks."""

        if not self.context:
            yield "No document context available."
            return

        prompt = self._get_answer_check_prompt(self._get_truncated_context(ANSWER_CHECK_CONTEXT_CHARS), challenge_question, user_answer)
        messages = [{"role": "user", "content": prompt}]

        emitted = False
        try:
            logging.info(f"CHECK_CALC_ANSWER_STREAM: Streaming with {MODEL_PRIMARY} + reasoning_effort=medium...")
            async for chunk in self._make_async_openai_streaming_call(
                messages=messages,
                model=MODEL_PRIMARY,
                max_tokens=8000,
                timeout=120,
                reasoning_effort="medium"
            ):
                emitted = True
                yield chunk
            logging.info("CHECK_CALC_ANSWER_STREAM: Streaming completed")
        except Exception as e:
            logging.error(f"CHECK_CALC_ANSWER_STREAM: {MODEL_PRIMARY} streaming failed: {e}")
            if emitted:
                yield "\n\n[Connection interrupted - please ask me to continue]"
            else:
                yield f"**Feedback:** I received your answer: {user_answer}. However, I'm having trouble processing the evaluation right now. Please try again in a moment."
