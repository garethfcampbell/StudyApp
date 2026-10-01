
import io
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
MODEL_PRIMARY = "gpt-6-luna"
MODEL_FALLBACK = "gpt-6-sol"
MODEL_FAST = "gpt-6-luna"

# First-choice model per feature. Generation / condensation features run on the
# fast, 20x cheaper model (benchmarked as faster too); chat, answer marking and
# the infographic formula review keep MODEL_PRIMARY. The fallback for a feature
# is always "the other" model.
SUMMARY_MODEL = MODEL_FAST
ESSAY_MODEL = MODEL_FAST        # essay-round generation (marking stays on MODEL_PRIMARY)
QUIZ_MODEL = MODEL_FAST
CALC_MODEL = MODEL_FAST         # calculation questions / worked examples (answer checking stays on MODEL_PRIMARY)
EXTRACTION_MODEL = MODEL_FAST   # equation list / exam question extraction
BRIEF_MODEL = MODEL_FAST        # infographic text brief


def _fallback_for(model):
    """The other model: luna falls back to sol and sol to luna."""
    return MODEL_FALLBACK if model == MODEL_PRIMARY else MODEL_PRIMARY

# How many times the PRIMARY model is attempted (timeouts, connection errors,
# rate limits, 5xx, empty responses) before a feature falls back to
# MODEL_FALLBACK. Client errors (4xx) and content filtering fail fast.
PRIMARY_MAX_ATTEMPTS = 3

# The executive summary is retrieval and condensation, not reasoning: low effort
# roughly halves its latency (benchmarked 11 s -> 6.5 s on gpt-6-sol).
SUMMARY_REASONING_EFFORT = "low"
# Document-type classification is a short recognition task: low effort.
CLASSIFIER_REASONING_EFFORT = "medium"
# Chat and the multiple-choice quiz are explanation / retrieval tasks: low effort
# halves chat's time to first token and takes a third off quiz generation.
CHAT_REASONING_EFFORT = "low"
QUIZ_REASONING_EFFORT = "low"   # quiz generation (faster first question)
# Essay-round generation and the infographic brief are content generation /
# condensation: low effort (~40% faster). Essay MARKING keeps the medium default.
ESSAY_REASONING_EFFORT = "low"   # essay ROUND generation (marking keeps the medium default)
INFOGRAPHIC_BRIEF_REASONING_EFFORT = "medium"
PRIMARY_RETRY_DELAY = 2  # seconds; multiplied by the attempt number


def _model_attempt_order():
    """Models to try in order: the primary several times, then the fallback once."""
    return [MODEL_PRIMARY] * PRIMARY_MAX_ATTEMPTS + [MODEL_FALLBACK]


class _RetryableEmptyResponse(ValueError):
    """The API returned no content without a content-filter or length reason."""

# Models in this family do not support system messages (all messages must be
# combined into a single user message), use max_completion_tokens instead of
# max_tokens, and reject non-default temperature values.
NO_SYSTEM_MESSAGE_MODELS = ("gpt-6-sol", "gpt-5", "gpt-5-mini", "gpt-6-luna", "gpt-5.4")

# Reasoning effort applied to every reasoning-model call unless a feature passes
# its own value explicitly (the calculation features pass "medium" themselves).
DEFAULT_REASONING_EFFORT = "medium"

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

    def _build_feature_messages(self, persona, prompt, guidance=None):
        """Build the standard system+user message pair for a document feature.
        `guidance` overrides the material-type guidance (pass "" to omit it)."""
        truncated_context = self._get_truncated_context()
        material_guidance = self._material_guidance() if guidance is None else guidance
        # The document goes FIRST so every feature's prompt shares the same long
        # prefix: OpenAI prompt caching then bills those input tokens at the
        # cached rate (about a tenth) for later features on the same document.
        return [
            {"role": "system", "content": f"{self._material_label()}:\n{truncated_context}\n\n{persona}\n\n{material_guidance}".rstrip()},
            {"role": "user", "content": prompt},
        ]

    # ---------------- essay practice: material-type handling ----------------
    def _essay_material_guidance(self):
        """System-level guidance for essay practice. Unlike other features, essay
        practice MUST provide model answers, so the generic 'do not answer the
        questions' rule for exam papers is replaced with exam-paper working."""
        if self.is_research_article():
            return self._RESEARCH_GUIDANCE
        if self.is_question_set():
            return (
                f"MATERIAL TYPE: The uploaded document is {self._material_description()}. It contains "
                "questions rather than notes. For essay practice, work through its ORIGINAL essay-style "
                "(discursive) questions - those asking students to explain, discuss, evaluate, compare or "
                "assess - one per round and in the order they appear, providing full suggested answers "
                "based on the finance concepts each question tests. Skip purely numerical calculation "
                "questions.\n\n"
            )
        return ""

    def _essay_grounding_text(self):
        """Grounding clause shared by the essay generation and marking prompts."""
        if self.is_question_set():
            return (
                "GROUNDING RULE (MOST IMPORTANT): the study material is a past exam paper / question sheet, "
                "so it contains questions, not notes. Base everything on the topics its questions test, "
                "drawing on the standard module knowledge those questions assume, and cite the question "
                "you are working from, e.g. \"(Question 3)\". Do NOT introduce topics the paper does not "
                "examine. Additional literature and real-world examples may go beyond the paper but must "
                "connect directly to a concept it examines."
            )
        return (
            "GROUNDING RULE (MOST IMPORTANT): every question, and every point in a suggested answer, must "
            "come from the study material - the concepts, theories, arguments and evidence it actually "
            "contains. Cite where each point comes from using the page/slide markers in the material, e.g. "
            "\"(see Slide 12)\". Do NOT set questions on topics the material does not cover. Additional "
            "literature and real-world examples are the ONLY things that may go beyond the material, and "
            "they must connect directly to a concept in it."
        )

    def _essay_notice_text(self):
        """Prominent notice for exam-paper essay practice: the app sees only the
        paper, so students must check suggested answers against their own notes."""
        if not self.is_question_set():
            return ""
        return (
            "**PLEASE NOTE - CHECK AGAINST YOUR LECTURE NOTES:** this practice is based only on the "
            "exam paper you uploaded. The app cannot see your lecture notes or the module's model "
            "answers, so the suggested answers below draw on general finance knowledge. Refer to your "
            "own lecture notes to confirm that the topics, theories and examples used are consistent "
            "with what has been covered in your module, and prioritise what your lecturer taught.\n\n"
        )

    def _essay_mode_block(self):
        """Extra instructions for the generation prompt when the material is an exam paper."""
        if not self.is_question_set():
            return ""
        return (
            "\nEXAM PAPER MODE: The MODEL QUESTION must be the NEXT original essay-style question from the "
            "paper that does not appear in the already-used list, quoted verbatim with its question number "
            "and marks (e.g. \"Question 4 (b) [15 marks]: ...\"). Its suggested answer is a full answer to "
            "that original question. YOUR QUESTION must then be a NEW question that is similar in topic, "
            "style, marks and difficulty to that original - the kind of variation an examiner might set "
            "next year - but not a copy or a trivial rewording. If EVERY essay-style question in the paper "
            "already appears in the used list, say so in one line, then set both the model question and "
            "your question as new questions on topics the paper examines. The PLEASE NOTE paragraph shown "
            "in the output structure MUST be reproduced word for word, directly under the main heading, "
            "before the model question.\n"
        )

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
            api_args["reasoning_effort"] = reasoning_effort or DEFAULT_REASONING_EFFORT
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

        # The SDK client already retries transient errors (max_retries=2) inside
        # each call. On top of that, the PRIMARY model is attempted several times
        # - with a longer timeout each time, since reasoning responses can be slow
        # - before the caller falls back to the secondary model. Client errors
        # (4xx) and content filtering are never retried.
        attempts = PRIMARY_MAX_ATTEMPTS if model in (MODEL_PRIMARY, MODEL_FAST) else 1
        last_error = None
        for attempt in range(1, attempts + 1):
            attempt_timeout = min(timeout * (1 + 0.5 * (attempt - 1)), 300)
            reason = None
            try:
                response = await asyncio.wait_for(
                    client.chat.completions.create(**api_args),
                    timeout=attempt_timeout
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
                    raise _RetryableEmptyResponse("Async OpenAI service returned an empty response.")

                if attempt > 1:
                    logging.info(f"Async OpenAI call to {model} succeeded on attempt {attempt}/{attempts}")
                return content

            except asyncio.TimeoutError as e:
                last_error, reason = e, f"timed out after {attempt_timeout:.0f}s"
            except (openai.APIConnectionError, openai.RateLimitError) as e:
                # Retryable classes (APITimeoutError subclasses APIConnectionError)
                last_error, reason = e, f"{type(e).__name__}: {e}"
            except openai.APIStatusError as e:
                if e.status_code < 500:
                    logging.error(f"Async OpenAI call to {model} failed with non-retryable API error {e.status_code}: {type(e).__name__} - {e}")
                    raise
                last_error, reason = e, f"server error {e.status_code}: {type(e).__name__}"
            except _RetryableEmptyResponse as e:
                last_error, reason = ValueError(str(e)), "empty response"
            except Exception as e:
                # Anything else (parsing errors, content filter, length) - fail fast.
                logging.error(f"Async OpenAI call to {model} failed with non-retryable error: {type(e).__name__} - {e}")
                raise

            if attempt < attempts:
                delay = PRIMARY_RETRY_DELAY * attempt
                logging.warning(f"Async OpenAI call to {model} {reason} (attempt {attempt}/{attempts}); retrying in {delay}s")
                await asyncio.sleep(delay)
            else:
                logging.error(f"Async OpenAI call to {model} {reason} (attempt {attempt}/{attempts}); giving up on this model")
        raise last_error

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

        async def _guarded(raw_stream):
            """Yield chunks; once the model has sent a finish_reason the answer is
            complete, so a transport error while the connection closes must not
            be reported to the student as an interruption."""
            finished = False
            try:
                async for chunk in raw_stream:
                    if chunk.choices and chunk.choices[0].finish_reason:
                        finished = True
                    yield chunk
            except Exception as e:
                if finished:
                    logging.warning(f"Streaming ({model}): ignoring error after completion: {type(e).__name__}: {e}")
                    return
                raise

        lead_buffer = ""
        lead_done = False
        async for chunk in _guarded(stream):
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

    def _stream_with_fallback_for(self, primary, stream_factory, label, failure_message):
        """_stream_with_fallback with a feature-specific first-choice model."""
        return self._stream_with_fallback(stream_factory, label, failure_message,
                                          primary=primary, fallback=_fallback_for(primary))

    async def _stream_with_fallback(self, stream_factory, label, failure_message, primary=None, fallback=None):
        """Run a primary-model stream with a fallback to the secondary model.

        The fallback is only attempted if the primary stream failed BEFORE any
        chunk was emitted. If output has already reached the user, restarting
        from scratch would show a truncated answer followed by a full second
        answer, so instead we log the error and emit a short interruption
        marker.
        """
        primary = primary or MODEL_PRIMARY
        fallback = fallback or _fallback_for(primary)
        emitted = False
        primary_error = None
        for attempt in range(1, PRIMARY_MAX_ATTEMPTS + 1):
            try:
                async for chunk in stream_factory(primary):
                    emitted = True
                    yield chunk
                return
            except Exception as e:
                primary_error = e
                if emitted:
                    logging.error(f"{label}: {primary} failed mid-stream after output was emitted: {e}")
                    yield "\n\n[Connection interrupted - please ask me to continue]"
                    return
                # A 4xx (other than rate limiting) will not succeed on retry
                client_error = (isinstance(e, openai.APIStatusError) and e.status_code < 500
                                and not isinstance(e, openai.RateLimitError))
                if attempt < PRIMARY_MAX_ATTEMPTS and not client_error:
                    delay = PRIMARY_RETRY_DELAY * attempt
                    logging.warning(f"{label}: {primary} failed before emitting output "
                                    f"(attempt {attempt}/{PRIMARY_MAX_ATTEMPTS}): {type(e).__name__}: {e}; retrying in {delay}s")
                    await asyncio.sleep(delay)
                    continue
                logging.error(f"{label}: {primary} failed before emitting output "
                              f"(attempt {attempt}/{PRIMARY_MAX_ATTEMPTS}): {type(e).__name__}: {e}; falling back to {fallback}")
                break

        try:
            async for chunk in stream_factory(fallback):
                emitted = True
                yield chunk
        except Exception as e:
            if emitted:
                logging.error(f"{label}: {fallback} failed mid-stream after output was emitted: {e}")
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
- Start with a single title line naming the subject of the material, then a one-line subtitle (no more than 15 words) saying in plain words what the topic is about, for a reader who is new to it.
- Then give 4-6 clearly titled sections (fewer, broader sections are better than many small ones). In each section give 3-4 bullet points, each a complete, self-explanatory sentence of no more than 16 words (one or two lines on a poster) - only the most important concepts, definitions and takeaways. Together they must give a short, clear summary that someone who has NOT read the notes can understand: briefly define a technical term the first time it appears, and prefer plain words. This is a visual poster, not notes: no detail beyond the key points.
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
            logging.info(f"INFOGRAPHIC: Summarizing lecture with {BRIEF_MODEL} for the image prompt...")
            summary = await self._make_async_openai_fallback_call(
                messages=messages, model=BRIEF_MODEL, temperature=0.2,
                max_tokens=4000, timeout=90, reasoning_effort=INFOGRAPHIC_BRIEF_REASONING_EFFORT
            )
        except Exception as primary_error:
            logging.error(f"INFOGRAPHIC: {BRIEF_MODEL} summarization failed: {primary_error}; trying {_fallback_for(BRIEF_MODEL)}")
            summary = await self._make_async_openai_fallback_call(
                messages=messages, model=_fallback_for(BRIEF_MODEL), temperature=0.2,
                max_tokens=4000, timeout=90, reasoning_effort=INFOGRAPHIC_BRIEF_REASONING_EFFORT
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

    async def generate_infographic_async(self, notes_brief=None):
        """Generate a one-page revision-guide infographic for the current
        document using the OpenAI images API. Returns a base64-encoded PNG.
        `notes_brief` may be supplied when it was pre-generated after upload."""
        if not self.context:
            raise ValueError("No document context set for infographic generation")

        # The images API accepts a far smaller prompt than the chat models, so
        # first condense the WHOLE lecture into that budget with a chat-model
        # summarization pass (a plain truncation would drop later sections).
        if notes_brief:
            logging.info("INFOGRAPHIC: using the pre-generated brief")
        else:
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
            "- CLEAN STYLE: the page background must be pure white (no colour tint, "
            "no gradient, no textured or coloured backdrop). Each section card sits "
            "on its OWN soft PASTEL tint, a different one per card, chosen from a "
            "harmonious set of very light, low-saturation pastels - for example pale "
            "sky blue (#EAF2FB), pale mint (#E9F7EF), pale peach (#FDF1E7), pale "
            "lavender (#F1ECFA), pale aqua (#E6F6F8), pale butter yellow (#FFF8E1). "
            "Use the tints in EXACTLY this order for cards 1 to 6 (card 1 sky blue, "
            "2 mint, 3 peach, 4 lavender, 5 aqua, 6 yellow), one tint per card, so "
            "neighbouring cards differ "
            "and the whole page feels calm and coordinated. Every tint must be flat "
            "and uniform, light enough that charcoal text is fully legible, and "
            "clearly pastel: never red or pink, never a saturated, dark or neon "
            "colour, never a gradient or texture. Cards have a hairline border a "
            "shade darker than their tint and a very soft shadow. Formula callouts "
            "and diagram boxes inside a card are white, so they contrast with the "
            "card.\n"
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
            "- Keep text short and clear: one bold title with a one-line subtitle, one "
            "heading per section, and 3-4 bullets per section, each a complete short "
            "sentence of one or two lines, set large enough to read easily. No "
            "paragraphs, no small print. The poster must make sense as a summary to "
            "someone who has not read the notes.\n"
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
        bad = self._blackout_region(image_b64)
        if bad:
            logging.warning(f"INFOGRAPHIC: generated image has a blacked-out area ({bad}); regenerating once")
            response = await client.images.generate(
                model=self.INFOGRAPHIC_MODEL, prompt=prompt, n=1, size="1024x1536", quality="high", timeout=300,
            )
            retry_b64 = response.data[0].b64_json if response.data else None
            if retry_b64 and not self._blackout_region(retry_b64):
                image_b64 = retry_b64

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

BE LENIENT. The poster is already a good draft; a correction costs time and money and can introduce new flaws, so report ONLY problems that would genuinely mislead a student or make part of the poster unusable. Report NOTHING about colours, tints, fonts, layout, spacing, photo size or style, and ignore single misspelled words in ordinary text.

Report as "major" ONLY these:
1. A FORMULA that is wrong: different symbols, subscripts or exponents from the brief, a missing or misplaced bracket, a fraction with the wrong numerator or denominator, visible LaTeX source (backslashes or braces), or a symbol so garbled it cannot be read. A formula that is correct but typeset in a slightly different style is NOT an issue.
2. A section of the brief that is MISSING entirely, or a card/area that is blank or solid black.
3. A numerical worked example, substituted numbers or a calculated answer (none are allowed), or a fact or formula that is not in the brief.
4. Text that is unreadable or so garbled that its meaning is lost (a heading, bullet or symbol key that cannot be understood).

Anything else you notice is "minor" and will be logged but NOT corrected. When in doubt, call it minor.

OUTPUT: respond with ONLY a JSON object, no other text:
{{"ok": true}} if there are no issues, otherwise
{{"ok": false, "issues": [{{"severity": "major" | "minor", "location": "<section heading or area of the page>", "problem": "<what is wrong>", "correction": "<exactly what it should show instead>", "box": [x0, y0, x1, y1]}}]}}
"box" is the bounding box of the WHOLE card or panel that contains the problem, as fractions of the poster's width and height measured from the top-left corner (e.g. [0.02, 0.70, 0.66, 0.96]). Give it for every issue and be generous so the box fully contains the card; use [0, 0, 1, 1] only for a page-wide problem.
List at most 6 issues, most important first. Be precise and literal - do not invent problems. "ok" is true when there are no MAJOR issues.

REVISION BRIEF:
{notes_brief}"""
        # Main review and the narrow spell-check pass are independent: run them
        # concurrently (saves ~15 s per infographic) and merge afterwards.
        review_task = asyncio.create_task(self._vision_json(review_prompt, [image_b64], "INFOGRAPHIC CHECK"))
        spell_task = asyncio.create_task(self._spellcheck_infographic(image_b64, notes_brief))
        result = await review_task
        issues = [i for i in (result.get("issues") or []) if isinstance(i, dict)][:8]
        try:
            spell_issues = await spell_task
            seen = {str(i.get("problem", "")).lower() for i in issues}
            issues += [t for t in spell_issues if str(t.get("problem", "")).lower() not in seen]
        except Exception as e:
            logging.error(f"INFOGRAPHIC CHECK: spell-check pass unavailable: {e}")
        majors = [i for i in issues if str(i.get("severity", "major")).lower() == "major"]
        minors = [i for i in issues if i not in majors]
        for m in minors:
            logging.info(f"INFOGRAPHIC CHECK: minor (not corrected) - {m.get('location')}: {m.get('problem')}")
        result = {"ok": not majors, "issues": majors, "minor": minors}
        logging.info(f"INFOGRAPHIC CHECK: ok={result['ok']}, {len(majors)} major / {len(minors)} minor")
        return result

    async def _vision_json(self, prompt_text, images_b64, label):
        """Send text + one or more PNGs to the vision-capable chat model (primary then
        fallback) and return the parsed JSON object. Raises if both models fail."""
        client = _get_async_openai_client()
        content = [{"type": "text", "text": prompt_text}]
        content += [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b}", "detail": "high"}}
                    for b in images_b64]
        messages = [{"role": "user", "content": content}]
        last_error = None
        order = _model_attempt_order()
        for attempt, model in enumerate(order, 1):
            try:
                api_args = {
                    "model": model,
                    "messages": messages,
                    "max_completion_tokens": 4000,
                    "response_format": {"type": "json_object"},
                    "reasoning_effort": DEFAULT_REASONING_EFFORT,
                }
                response = await asyncio.wait_for(client.chat.completions.create(**api_args), timeout=180)
                content_text = (response.choices[0].message.content or "").strip()
                result = json.loads(content_text)
                if not isinstance(result, dict):
                    raise ValueError("Response was not a JSON object")
                logging.info(f"{label} ({model}): response received")
                return result
            except Exception as e:
                last_error = e
                if attempt < len(order):
                    delay = PRIMARY_RETRY_DELAY * attempt if model == MODEL_PRIMARY else 0
                    logging.warning(f"{label}: {model} failed (attempt {attempt}/{len(order)}): {type(e).__name__}: {e}; "
                                    f"next: {order[attempt]}")
                    if delay:
                        await asyncio.sleep(delay)
                else:
                    logging.error(f"{label}: {model} failed on the final attempt: {type(e).__name__}: {e}")
        raise last_error if last_error else RuntimeError(f"{label} failed")

    # ---------------- spell-check pass (adapted from the daily infographic pipelines) ----------------
    @staticmethod
    def _image_tiles(image_bytes, cols=2, rows=6, overlap=30, scale=3):
        """Overlapping tiles of the poster, enlarged so small labels survive the API's
        downscaling. Returns base64 PNG strings, row by row from the top left."""
        import base64
        from PIL import Image
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        w, h = img.size
        out = []
        for r in range(rows):
            for c in range(cols):
                box = (max(0, c * w // cols - overlap), max(0, r * h // rows - overlap),
                       min(w, (c + 1) * w // cols + overlap), min(h, (r + 1) * h // rows + overlap))
                tile = img.crop(box).resize(((box[2] - box[0]) * scale, (box[3] - box[1]) * scale), Image.LANCZOS)
                buf = io.BytesIO()
                tile.save(buf, format="PNG")
                out.append(base64.b64encode(buf.getvalue()).decode())
        return out

    async def _spellcheck_infographic(self, image_b64, notes_brief, already=()):
        """Narrow vision pass: find misspelled or garbled words (and mangled formulas) by
        reading enlarged tiles letter by letter. Returns issues in the review format."""
        import base64
        image_bytes = base64.b64decode(image_b64)
        tiles = self._image_tiles(image_bytes)
        prompt_text = f"""This revision poster was generated by an image model, which sometimes garbles individual words - most often in short labels: headings, bullet text, symbol keys, chart and axis labels, diagram boxes - and occasionally a formula symbol. Your ONLY job is to find misspelled or garbled words and garbled formula symbols.

The first image is the whole poster; the remaining images are enlarged tiles of it, row by row from the top left, overlapping slightly.

Step 1: for every short label (heading, bullet, symbol key, chart or axis label, diagram box, caption) write it out LETTER BY LETTER separated by hyphens, e.g. V-o-l-a-t-i-l-i-t-y, reading the glyphs as drawn rather than the word you expect, checking especially for doubled, dropped or swapped letters; transcribe longer text exactly as printed, spelling out letter by letter any word you are not certain of.

Step 2: compare every transcribed word with the REVISION BRIEF below. BE LENIENT: report ONLY words that are garbled or truncated badly enough that a reader could not tell what word was intended, and formula symbols that are visibly mangled (a broken glyph, a stray character, a subscript rendered as a normal letter). Do NOT report a word with a single wrong, dropped or doubled letter when the intended word is still obvious.

Do NOT report differences of layout, hyphenation at a line break, capitalisation, curly versus straight quotes, a missing or doubled space, or a hyphen in place of a space: those are not misspellings. Report only words you are certain are wrong.

Return ONLY a JSON object: {{"typos": [{{"printed": "<as printed>", "expected": "<correct word>", "where": "<section heading>", "box": [x0, y0, x1, y1]}}]}} where "box" is the bounding box of the WHOLE card or panel containing the word, as fractions of the WHOLE poster's width and height (use the first image), generous enough to contain the card.

REVISION BRIEF:
{notes_brief}"""
        data = await self._vision_json(prompt_text, [image_b64] + tiles, "INFOGRAPHIC SPELLCHECK")
        issues, seen = [], {str(i.get("problem", "")).lower() for i in already}
        for t in data.get("typos", []) or []:
            if not isinstance(t, dict):
                continue
            printed, expected = str(t.get("printed", "")).strip(), str(t.get("expected", "")).strip()
            if not printed or printed.lower() == expected.lower():
                continue
            key = f'"{printed}"'.lower()
            if any(key in s_ for s_ in seen):
                continue
            seen.add(key)
            issues.append({
                "severity": "major",
                "location": t.get("where") or "poster",
                "problem": f'"{printed}" is printed where "{expected}" is intended',
                "correction": f'Print "{expected}" exactly, spelled correctly, in the same style and size',
                "box": t.get("box"),
            })
        logging.info(f"INFOGRAPHIC SPELLCHECK: {len(issues)} typo(s)")
        return issues[:6]

    # ---------------- masked repair ----------------
    @staticmethod
    def _repair_boxes(issues, pad=0.02):
        """Bounding boxes for a masked repair, or None if any issue lacks a usable box."""
        boxes = []
        for i in issues:
            b = i.get("box") if isinstance(i, dict) else None
            if not (isinstance(b, list) and len(b) == 4):
                return None
            try:
                x0, y0, x1, y1 = [max(0.0, min(1.0, float(v))) for v in b]
            except (TypeError, ValueError):
                return None
            if x1 <= x0 or y1 <= y0:
                return None
            boxes.append([max(0.0, x0 - pad), max(0.0, y0 - pad), min(1.0, x1 + pad), min(1.0, y1 + pad)])
        return boxes or None

    @staticmethod
    def _blackout_region(image_b64, boxes=None, threshold=0.6):
        """Detect a solid black (or near-black) area: the signature of a failed
        image edit that repainted a card with nothing. Checks each box (fractions
        of the poster) when given, otherwise scans the whole poster in a coarse
        grid. Returns a description of the first offending region, or None."""
        import base64
        from PIL import Image
        img = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("L")
        w, h = img.size
        regions = []
        if boxes:
            regions = [(int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)) for x0, y0, x1, y1 in boxes]
        else:
            cols, rows = 8, 12  # cells of ~1/8 x 1/12 of the page: any near-black cell is a defect on a white poster
            regions = [(c * w // cols, r * h // rows, (c + 1) * w // cols, (r + 1) * h // rows)
                       for r in range(rows) for c in range(cols)]
        for (x0, y0, x1, y1) in regions:
            if x1 - x0 < 8 or y1 - y0 < 8:
                continue
            hist = img.crop((x0, y0, x1, y1)).histogram()
            total = float(sum(hist)) or 1.0
            dark = sum(hist[:32]) / total  # luminance below 32/255
            if dark >= threshold:
                return f"{int(dark * 100)}% near-black pixels in region x={x0}-{x1}, y={y0}-{y1}"
        return None

    @staticmethod
    def _make_mask_png(image_bytes, boxes):
        """PNG mask the size of the poster: transparent where the edit may happen, opaque
        elsewhere (the images edit endpoint repaints only transparent pixels)."""
        from PIL import Image, ImageDraw
        img = Image.open(io.BytesIO(image_bytes))
        w, h = img.size
        mask = Image.new("RGBA", (w, h), (0, 0, 0, 255))
        draw = ImageDraw.Draw(mask)
        for x0, y0, x1, y1 in boxes:
            draw.rectangle([int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)], fill=(0, 0, 0, 0))
        buf = io.BytesIO()
        mask.save(buf, format="PNG")
        return buf.getvalue()

    _FIX_RULES = (
        "RULES WHILE EDITING: formulas must be typeset with complete accuracy - "
        "exactly the symbols, subscripts, exponents and bracket placement given in "
        "the correction, with nothing dropped, added or rearranged - as proper "
        "LaTeX-style mathematics (Computer Modern look): stacked fractions with a "
        "horizontal bar, true sub/superscripts, bars and hats over letters, Greek "
        "glyphs, a large sigma for sums; never a slash for division, never symbols "
        "spelled as words, never visible LaTeX source. Text must be "
        "spelled correctly and fully legible. Show no numerical worked examples or "
        "calculated answers. The page must stay pure white; each card keeps its "
        "own soft light pastel tint (different per card - pale blue, mint, peach, "
        "lavender, yellow, aqua), never red, pink, saturated or dark fills; the "
        "only accent is deep red used for "
        "headings, thin rules, arrows, outlines and small badges, and formula "
        "boxes are white with a thin red left border. Keep one clean sans-serif "
        "font family. Photos may only be small rounded vignettes (about one fifth "
        "of a card), at most 2 on the page plus a small hero image, all of "
        "different subjects, with no screens, charts or text inside them - "
        "shrink, replace or remove a photo if a correction asks for it and give "
        "the space to the diagram, formula or white space. Do not add any "
        "content that is not in the brief.\n\n"
    )

    # Whether the image model / SDK accept input_fidelity. Learned on first use:
    # older SDKs raise TypeError, and some image models (gpt-image-2.5-sunburst)
    # reject it with a 400, in which case we stop sending it.
    _INPUT_FIDELITY_SUPPORTED = True

    async def _images_edit(self, edit_args):
        """images.edit with input_fidelity="high" when the SDK and model accept it."""
        client = _get_async_openai_client()
        if TutorAI._INPUT_FIDELITY_SUPPORTED:
            try:
                return await client.images.edit(input_fidelity="high", **edit_args)
            except (TypeError, openai.BadRequestError) as e:
                if "input_fidelity" not in str(e):
                    raise
                TutorAI._INPUT_FIDELITY_SUPPORTED = False
                logging.info("INFOGRAPHIC FIX: input_fidelity not accepted by the SDK/model; retrying without it")
        return await client.images.edit(**edit_args)

    async def _fix_infographic(self, image_b64, issues, notes_brief):
        """Correct the listed issues. When every issue carries a bounding box, repaint ONLY
        those cards through a masked edit (everything outside the mask is preserved
        pixel for pixel); otherwise fall back to a whole-image edit that is asked to
        change only the affected areas. Returns the corrected base64 PNG."""
        import base64
        corrections = "\n".join(
            f"{i + 1}. {issue.get('location', 'page')}: {issue.get('problem', '')} -> "
            f"Correct to: {issue.get('correction', '')}"
            for i, issue in enumerate(issues)
        )
        image_bytes = base64.b64decode(image_b64)
        base_args = dict(model=self.INFOGRAPHIC_MODEL, n=1, size="1024x1536", quality="high", timeout=300)

        boxes = self._repair_boxes(issues)
        if boxes:
            repair_prompt = (
                "REPAIR of an existing revision poster. The attached image is the finished "
                "poster and the mask marks the ONLY area you may change. Repaint that area "
                "so that it corrects these problems found by a reviewer:\n\n"
                f"CORRECTIONS:\n{corrections}\n\n"
                "Everything outside the masked area must stay exactly as it is. Inside the "
                "area keep the same card tint, heading style, fonts, text size and layout as "
                "the rest of the poster, and keep every correct element of that card.\n\n"
                + self._FIX_RULES
                + f"REVISION BRIEF (source of truth):\n{notes_brief}"
            )
            logging.info(f"INFOGRAPHIC FIX: masked repair of {len(boxes)} card(s) for {len(issues)} issue(s): "
                         f"{[[round(v, 2) for v in b] for b in boxes]}")
            try:
                mask_bytes = self._make_mask_png(image_bytes, boxes)
                response = await self._images_edit(dict(
                    image=("infographic.png", image_bytes, "image/png"),
                    mask=("mask.png", mask_bytes, "image/png"),
                    prompt=repair_prompt[:32000],
                    **base_args,
                ))
                fixed_b64 = response.data[0].b64_json if response.data else None
                if fixed_b64:
                    bad = self._blackout_region(fixed_b64, boxes)
                    if bad:
                        logging.warning(f"INFOGRAPHIC FIX: masked repair blacked out a card ({bad}); "
                                        "discarding it and falling back to whole-image edit")
                    else:
                        logging.info(f"INFOGRAPHIC FIX: masked repair received ({len(fixed_b64)} base64 chars)")
                        return fixed_b64
                else:
                    logging.warning("INFOGRAPHIC FIX: masked repair returned no image; falling back to whole-image edit")
            except Exception as e:
                logging.warning(f"INFOGRAPHIC FIX: masked repair failed ({str(e)[:200]}); falling back to whole-image edit")

        fix_prompt = (
            "Edit this revision infographic. Keep the overall layout, section order, "
            "photographs, diagrams and all correct text exactly as they are. Apply ONLY "
            "the corrections listed below, redrawing just the affected areas.\n\n"
            f"CORRECTIONS:\n{corrections}\n\n"
            + self._FIX_RULES
            + f"REVISION BRIEF (source of truth):\n{notes_brief}"
        )
        logging.info(f"INFOGRAPHIC FIX: whole-image edit for {len(issues)} issue(s)")
        response = await self._images_edit(dict(
            image=("infographic.png", image_bytes, "image/png"),
            prompt=fix_prompt,
            **base_args,
        ))
        fixed_b64 = response.data[0].b64_json if response.data else None
        if not fixed_b64:
            raise ValueError("Image edit returned no image data")
        bad = self._blackout_region(fixed_b64)
        if bad:
            raise ValueError(f"Image edit produced a blacked-out area ({bad}); keeping the previous image")
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
                timeout=60,
                reasoning_effort=CHAT_REASONING_EFFORT
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
                    timeout=60,
                    reasoning_effort=CHAT_REASONING_EFFORT
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
                messages=messages, model=model, temperature=0.7, max_tokens=15000, timeout=60,
                reasoning_effort=CHAT_REASONING_EFFORT
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
                logging.info(f"ASYNC SUMMARY: Trying {SUMMARY_MODEL} for executive summary generation...")
                result = await self._make_async_openai_fallback_call(
                    messages=messages, model=SUMMARY_MODEL, temperature=0.2, max_tokens=15000, timeout=60,
                    reasoning_effort=SUMMARY_REASONING_EFFORT
                )
                if result and result.strip():
                    logging.info(f"ASYNC SUMMARY: {SUMMARY_MODEL} succeeded")
                    return _normalize_study_formatting(_strip_code_fences(result))
                raise ValueError(f"{SUMMARY_MODEL} returned an empty response")
            except Exception as mini_error:
                logging.error(f"ASYNC SUMMARY: {SUMMARY_MODEL} failed: {mini_error}")

            # Fallback model
            try:
                logging.info(f"ASYNC SUMMARY: Trying {_fallback_for(SUMMARY_MODEL)} fallback...")
                result = await self._make_async_openai_fallback_call(
                    messages=messages, model=_fallback_for(SUMMARY_MODEL), temperature=0.2, max_tokens=15000, timeout=60,
                    reasoning_effort=SUMMARY_REASONING_EFFORT
                )
                logging.info(f"ASYNC SUMMARY: {_fallback_for(SUMMARY_MODEL)} fallback succeeded")
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
                    messages=messages, model=model, temperature=0.2, max_tokens=15000, timeout=90,
                    reasoning_effort=SUMMARY_REASONING_EFFORT
                )

            async for chunk in _normalize_study_stream(self._stream_with_fallback_for(SUMMARY_MODEL, 
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

Print the template below EXACTLY: the headings, the order, the blank lines and the closing question. Replace every placeholder in angle or square brackets with your own content. The CONTENT RULES after the template are instructions to you - they must NEVER appear in the output.

***OVERVIEW***

<overview paragraph>

***KEY CONCEPTS***

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

Would you like to explore any of these topics in more detail?

### CONTENT RULES (never print these)

- <overview paragraph>: summarise the main subject/topic of the material (for an exam paper or question sheet, the areas its questions cover; for a research article, what it investigates and finds), in a professional, academic tone suitable for quick review before an exam.
- KEY CONCEPTS: the most important concepts with brief definitions, grouped into 5-6 logical categories (5 minimum, 6 maximum, each with 3-5 concepts), numbering the categories in sequence. Each concept is on its own line in the form "- *Concept:* brief explanation".
- The closing question ("Would you like to explore any of these topics in more detail?") is printed word for word as the last line.

            """

    async def generate_essay_question_async(self, used_questions=None):

        if not self.context:
            return "No study material available to create essay questions from."

        try:
            messages = self._build_feature_messages(
                "You are an experienced QUB Finance examiner who sets short qualitative essay questions, writes model answers grounded in the lecture material, and coaches students on how to answer for high marks.",
                self._get_essay_prompt(used_questions),
                guidance=self._essay_material_guidance()
            )

            logging.info(f"ASYNC ESSAY: Context length: {len(self._get_truncated_context())} characters")

            # Try the primary model
            try:
                logging.info(f"ASYNC ESSAY: Trying {ESSAY_MODEL} for essay generation...")
                result = await self._make_async_openai_fallback_call(
                    messages=messages, model=ESSAY_MODEL, temperature=0.4, max_tokens=15000, timeout=150,
                    reasoning_effort=ESSAY_REASONING_EFFORT
                )
                if result and result.strip():
                    logging.info(f"ASYNC ESSAY: {ESSAY_MODEL} succeeded")
                    return _normalize_study_formatting(_strip_code_fences(result), mode='essay')
                raise ValueError(f"{ESSAY_MODEL} returned an empty response")
            except Exception as mini_error:
                logging.error(f"ASYNC ESSAY: {ESSAY_MODEL} failed: {mini_error}")

            # Fallback model
            try:
                logging.info(f"ASYNC ESSAY: Trying {_fallback_for(ESSAY_MODEL)} fallback...")
                result = await self._make_async_openai_fallback_call(
                    messages=messages, model=_fallback_for(ESSAY_MODEL), temperature=0.4, max_tokens=15000, timeout=60,
                    reasoning_effort=ESSAY_REASONING_EFFORT
                )
                logging.info(f"ASYNC ESSAY: {_fallback_for(ESSAY_MODEL)} fallback succeeded")
                return _normalize_study_formatting(_strip_code_fences(result), mode='essay')
            except Exception as nano_error:
                logging.error(f"ASYNC ESSAY: All models failed: {nano_error}")
                return "I'm having trouble generating an essay question right now. The document appears to be loaded successfully, but there may be a temporary issue with the AI service. Please try again in a moment or use the chat to ask specific questions about your document."

        except Exception as e:
            logging.error(f"ASYNC ESSAY: Critical error in async essay generation: {e}")
            return f"Critical error in essay generation: {str(e)}"

    async def generate_essay_question_stream_async(self, used_questions=None):
        """Streaming version of generate_essay_question_async. Yields text chunks."""
        if not self.context:
            yield "No study material available to create essay questions from."
            return

        try:
            messages = self._build_feature_messages(
                "You are an experienced QUB Finance examiner who sets short qualitative essay questions, writes model answers grounded in the lecture material, and coaches students on how to answer for high marks.",
                self._get_essay_prompt(used_questions),
                guidance=self._essay_material_guidance()
            )

            def factory(model):
                return self._make_async_openai_streaming_call(
                    messages=messages, model=model, temperature=0.4, max_tokens=15000, timeout=150,
                    reasoning_effort=ESSAY_REASONING_EFFORT
                )

            async for chunk in _normalize_study_stream(self._stream_with_fallback_for(ESSAY_MODEL, 
                factory,
                "STREAM ESSAY",
                "I'm having trouble generating an essay question right now. Please try again in a moment."
            ), mode='essay'):
                yield chunk
        except Exception as e:
            logging.error(f"STREAM ESSAY: Critical error: {e}")
            yield f"Critical error in essay generation: {str(e)}"

    def _get_essay_prompt(self, used_questions=None):
        """Return the essay-practice prompt for ONE round: a model question with a
        suggested answer, then a new question for the student to attempt."""
        used_block = ""
        if used_questions:
            listed = "\n".join(f"- {q}" for q in used_questions[-20:])
            used_block = ("\nQUESTIONS ALREADY USED IN THIS SESSION - do NOT repeat or closely paraphrase any of them, "
                          "as either the model question or the student's question; choose different topics or angles:\n"
                          + listed + "\n")
        return r"""

            CRITICAL FORMATTING REQUIREMENTS:
            - Use markdown formatting for emphasis: **bold text**, *italic text*
            - **ABSOLUTELY NO MATHEMATICAL NOTATION:** Do NOT use LaTeX formatting, mathematical symbols, or any notation:
              * NO dollar signs: $x$, $\delta$, $P_t$, etc.
              * NO backslash notation: \(x\), \[equation\], etc.
              * NO mathematical symbols: √, ∑, ∫, ≤, ≥, ≠, π, etc.
            - NEVER use HTML tags - only use markdown formatting
            - ONLY USE HYPHENS FOR BULLETS (-) - never use asterisks (*) or dots (•)
            - Each bullet point must be on its own line with consistent hyphen formatting
            - Use ONLY plain English words to describe mathematical concepts and symbols (this rule is about formulas and symbols, NOT numbers)
            - NUMBERS ARE ALWAYS DIGITS: write years, percentages, marks, question numbers and counts as digits - "1973", "60-69%", "[15 marks]", "Question 2" - NEVER as words ("nineteen seventy-three" is wrong)
            - CITATION FORMAT: "Author(s) (Year), Title" with the year in digits, e.g. "Black and Scholes (1973), The Pricing of Options and Corporate Liabilities"
            - Always respond in plain text with markdown formatting only

            TASK: Run ONE round of essay-question practice, based strictly on the study material provided.

            {{GROUNDING}}
{{MODE}}
            QUESTION STYLE: short qualitative exam questions of the kind set in QUB Finance examinations - answered in prose in roughly 15-25 minutes (about 300-500 words) - using command words such as "Explain", "Discuss", "Critically evaluate", "Compare and contrast", "To what extent", "Assess". They must ask for analysis, evaluation or application, not description alone, and be realistic in wording, scope and difficulty. For an exam paper or question sheet, base them on the discursive questions it contains and the topics it tests; for a research article, on its question, method, findings and implications.

            ADDITIONAL LITERATURE RULE: cite the foundational, peer-reviewed academic literature on the topic - the seminal journal articles that established the theory or the key empirical findings (e.g. Markowitz (1952), Journal of Finance; Fama (1970), Journal of Finance; Black and Scholes (1973), Journal of Political Economy; Jensen and Meckling (1976), Journal of Financial Economics). Do NOT cite textbooks (Hull, Brealey and Myers, Bodie Kane and Marcus, etc.) - students already have the lecture notes for that level. Cite only real works you are confident exist, giving author(s), year, title and journal. If not certain a specific article exists, describe the body of literature instead (e.g. "the empirical literature on post-earnings-announcement drift") rather than inventing a citation.
{{USED}}
---

### REQUIRED OUTPUT STRUCTURE

Print the template below EXACTLY: the bold labels, the order, the blank lines and the closing sentence. Replace every placeholder written in angle brackets <like this> with your own content. The CONTENT RULES after the template are instructions to you - they must NEVER appear in the output, and nothing may follow a label on its line except what the template shows.

***ESSAY QUESTION PRACTICE***
{{NOTICE}}
**MODEL QUESTION**

*<model question>*

**Suggested answer**

- <point 1>
- <point 2>
- <point 3>
- <point 4>
- <point 5>
- <further points if needed, 8 at most>

**Additional literature**

- <source 1>
- <source 2>
- <Google Scholar search suggestion>

**Real-world examples**

- <example 1>
- <example 2>

**Why this answer scores highly**

<1-2 sentences>

**YOUR QUESTION:** <student's question>

**Hints**

- <hint 1>
- <hint 2>

Write your answer in the box below (aim for 300-500 words - bullet points are fine), then click **Check Answer**. I will mark it against the QUB Conceptual Equivalents Scale and show you a suggested answer.

### CONTENT RULES (never print these)

- <model question>: one short qualitative exam question on a major topic of the material, worded exactly as it would appear on an exam paper, ending with its marks in square brackets. In EXAM PAPER MODE, the next original question from the paper, quoted verbatim.
- Suggested answer points: 5-8 bullets giving the points a First-class answer would make, in a sensible order (define, apply, evaluate, conclude). Each bullet is one or two sentences, grounded in the material, with a citation such as "(see Slide 12)" for lecture notes or "(Question 3)" for an exam paper.
- Additional literature: 2 bullets, each a foundational peer-reviewed journal article (not a textbook) with one sentence on the point it supports and where in the answer to use it. The third bullet encourages the student to search Google Scholar for recent academic work on the topic and gives a specific search phrase, e.g. "Search Google Scholar for recent papers on 'post-earnings-announcement drift' (2015 onwards) to add up-to-date evidence to this answer".
- Real-world examples: 2 bullets, each a concrete example (named company, market, event, policy episode or crisis) that is NOT already used in the study material - choose additional examples the lecturer did not cover, so the student can show wider reading - with one sentence on how it strengthens the answer.
- Why this answer scores highly: 1-2 sentences explaining, with reference to the QUB Conceptual Equivalents Scale, what lifts it from a Lower Second (describing the concept) to an Upper Second (evaluating strengths and limitations with evidence) to a First (weighing competing perspectives with insight, well-chosen literature and examples).
- <student's question>: a DIFFERENT short qualitative exam question on a DIFFERENT major topic of the material - in EXAM PAPER MODE, a NEW question similar in topic, style, marks and difficulty to the model question - worded exactly as it would appear on an exam paper, on the SAME line as the bold label. Do not add a separate heading or repeat the words YOUR QUESTION.
- Hints: 2 bullets naming which parts of the material (slide/page citations for lecture notes; for an exam paper, the concepts the question tests) to draw on and what kind of analysis is expected. Do NOT give the answer.
- The closing sentence ("Write your answer in the box below ...") is printed word for word.

            RESPONSE FORMAT: Provide the formatted text directly - no JSON, no code blocks.""".replace("{{USED}}", used_block).replace("{{GROUNDING}}", self._essay_grounding_text()).replace("{{MODE}}", self._essay_mode_block()).replace("{{NOTICE}}", ("\n" + self._essay_notice_text()) if self._essay_notice_text() else "")

    def _get_essay_check_prompt(self, context_truncated, essay_round_text, user_answer):
        """Prompt for marking the student's answer to the question set in the last essay round."""
        return rf"""You are an experienced QUB Finance examiner marking a student's short qualitative exam answer. Be encouraging but honest and specific.

STUDY MATERIAL (the module content the question is based on):
{context_truncated}

THE PRACTICE ROUND SHOWN TO THE STUDENT (the question they were asked is the line containing "YOUR QUESTION:"):
{essay_round_text}

STUDENT'S ANSWER:
{user_answer}

MARK ONLY the question after "YOUR QUESTION:". Ignore the model question. If a PLEASE NOTE paragraph appears in the output structure below, reproduce it word for word directly under the main heading.

{self._essay_grounding_text()} When you say what should have been included, cite the relevant slide/page marker (lecture notes) or question number (exam paper). Cite only real, well-established sources; if unsure a source exists, describe the body of literature instead.

If the answer is empty, off-topic or says "I don't know", say so kindly, give the lowest band, and still provide the suggested answer.

CRITICAL FORMATTING REQUIREMENTS:
- Use markdown formatting for emphasis: **bold text**, *italic text*
- ABSOLUTELY NO MATHEMATICAL NOTATION (no LaTeX, no $ signs, no mathematical symbols) - describe formulas and symbols in plain English words
- NUMBERS ARE ALWAYS DIGITS: years, percentages, marks and counts as digits ("1973", "60-69%"), never as words; citations as "Author(s) (Year), Title"
- NEVER use HTML tags
- ONLY USE HYPHENS FOR BULLETS (-), each bullet on its own line
- Main heading in block capitals with bold and italic like ***THIS***; sub-headings in block capitals with bold like **THIS**

REQUIRED OUTPUT STRUCTURE: print the template below EXACTLY (bold labels, order, blank lines, closing sentence), replacing every placeholder in angle brackets <like this> with your own content. The CONTENT RULES after the template are instructions to you and must NEVER appear in the output.

***FEEDBACK ON YOUR ANSWER***

{self._essay_notice_text()}**GRADE BAND:** <band> - <justification>

**WHAT YOU DID WELL**

- <strength 1>
- <strength 2>
- <further strengths if needed, 4 at most>

**WHAT WAS MISSING OR WEAK**

- <gap 1>
- <gap 2>
- <gap 3>
- <further gaps if needed, 5 at most>

**SUGGESTED ANSWER**

- <point 1>
- <point 2>
- <point 3>
- <point 4>
- <point 5>
- <point 6>
- <further points if needed, 10 at most>

**TO REACH THE NEXT BAND**

- <literature 1>
- <literature 2>
- <Google Scholar search suggestion>
- <example 1>
- <example 2>
- <the single most important improvement>

Click **Next Question** for another essay question, or **End Practice** to return to the menu.

CONTENT RULES (never print these):
- <band>: one of First (70-100%), Upper Second (60-69%), Lower Second (50-59%), Third (40-49%), Fail (below 40%). <justification>: one or two sentences justifying the band against the QUB Conceptual Equivalents Scale (depth of critical analysis, insight, knowledge and understanding, coverage, use of sources).
- Strengths: 2-4 bullets, each naming a specific point or quality in the answer.
- Gaps: 3-5 bullets, each naming the specific concept, theory or argument from the material that should have been used (with a citation), or the analytical step that was skipped.
- Suggested answer: 6-10 bullets giving the points a First-class answer would make, in a sensible order (define, apply, evaluate, conclude), each grounded in the material with a citation.
- To reach the next band: 2-3 bullets of additional literature - foundational peer-reviewed journal articles (author(s), year, title, journal), NOT textbooks - that would strengthen the answer, each with the point it supports; then 1 bullet encouraging the student to search Google Scholar for recent academic work on the topic, with a specific search phrase; then 2-3 bullets of real-world examples NOT already used in the study material (additional cases the lecturer did not cover), each with how to use it; then 1 bullet with the single most important structural or analytical improvement.
- The closing sentence ("Click **Next Question** ...") is printed word for word."""

    async def check_essay_answer_stream_async(self, essay_round_text, user_answer):
        """Stream marking feedback for the student's essay answer (mirrors the calculation answer check)."""
        if not self.context:
            yield "No study material available to mark the answer against."
            return
        prompt = self._get_essay_check_prompt(self._get_truncated_context(ANSWER_CHECK_CONTEXT_CHARS), essay_round_text, user_answer)
        messages = [{"role": "user", "content": prompt}]

        def factory(model):
            return self._make_async_openai_streaming_call(
                messages=messages, model=model, temperature=0.3, max_tokens=12000, timeout=90
            )

        try:
            async for chunk in _normalize_study_stream(self._stream_with_fallback(
                factory,
                "ESSAY_ANSWER_STREAM",
                "I'm having trouble marking your answer right now. Please try again in a moment."
            ), mode='essay'):
                yield chunk
        except Exception as e:
            logging.error(f"ESSAY_ANSWER_STREAM: Critical error: {e}")
            yield f"Critical error while marking the answer: {str(e)}"

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
        return TutorAI._validate_quiz_questions(questions)

    @staticmethod
    def parse_quiz_partial(text):
        """Return the validated questions that are already COMPLETE in a quiz JSON
        response that is still being streamed. Scans the "questions" array for
        balanced {...} objects (string-aware) so the page can show the first
        question while the rest are still being generated."""
        if not text:
            return []
        start = text.find('"questions"')
        if start == -1:
            return []
        start = text.find('[', start)
        if start == -1:
            return []
        objects, depth, in_str, esc, obj_start = [], 0, False, False, None
        for i in range(start + 1, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == '\\':
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == '{':
                if depth == 0:
                    obj_start = i
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0 and obj_start is not None:
                    try:
                        objects.append(json.loads(text[obj_start:i + 1]))
                    except Exception:
                        pass
                    obj_start = None
            elif c == ']' and depth == 0:
                break
        return TutorAI._validate_quiz_questions(objects[:15])

    @staticmethod
    def _validate_quiz_questions(questions):
        """Validate/repair parsed quiz question dicts; malformed ones are skipped."""

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
                # (correct answer listed first) never reaches students. The
                # shuffle is seeded by the question text so the same question
                # parsed again later (from a growing stream) keeps its order.
                if correct_answer in options:
                    q["correct_answer"] = correct_answer
                random.Random(q["question"]).shuffle(options)
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
            prompt = self._get_quiz_prompt(context_truncated)
            messages = [{"role": "user", "content": prompt}]

            # Try the primary model
            try:
                logging.info(f"ASYNC QUIZ: Trying {QUIZ_MODEL} for retrieval quiz generation...")
                logging.info(f"ASYNC QUIZ: Context length: {len(context_truncated)} characters")

                result = await self._make_async_openai_fallback_call(
                    messages=messages,
                    model=QUIZ_MODEL,
                    response_format={"type": "json_object"},
                    temperature=0.3,
                    max_tokens=15000,
                    timeout=60,
                    reasoning_effort=QUIZ_REASONING_EFFORT
                )

                logging.info(f"ASYNC QUIZ: {QUIZ_MODEL} succeeded")
                valid_questions = self._parse_and_validate_quiz(result)
                logging.info(f"ASYNC QUIZ: Generated {len(valid_questions)} valid questions")
                return valid_questions

            except Exception as primary_error:
                logging.error(f"ASYNC QUIZ: {QUIZ_MODEL} failed: {primary_error}")
                # Fallback model
                try:
                    logging.info(f"ASYNC QUIZ: Trying {_fallback_for(QUIZ_MODEL)} fallback...")
                    fallback_result = await self._make_async_openai_fallback_call(
                        messages=messages,
                        model=_fallback_for(QUIZ_MODEL),
                        response_format={"type": "json_object"},
                        temperature=0.3,
                        reasoning_effort=QUIZ_REASONING_EFFORT,
                        max_tokens=15000,
                        timeout=60
                    )

                    logging.info(f"ASYNC QUIZ: {_fallback_for(QUIZ_MODEL)} fallback succeeded")
                    valid_questions = self._parse_and_validate_quiz(fallback_result)
                    logging.info(f"ASYNC QUIZ: Fallback generated {len(valid_questions)} valid questions")
                    return valid_questions

                except Exception as fallback_error:
                    logging.error(f"ASYNC QUIZ: Both async methods failed: {fallback_error}")
                    return []

        except Exception as e:
            logging.error(f"ASYNC QUIZ: Critical error in async quiz generation: {e}")
            return []

    def _get_quiz_prompt(self, context_truncated=None):
        """The multiple-choice quiz prompt (shared by the whole and streaming generators)."""
        if context_truncated is None:
            context_truncated = self._get_truncated_context()
        return rf"""Based on this study material, create exactly 15 simple multiple choice questions about the key concepts. Do NOT include any mathematical equations or formulas.

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

    async def generate_retrieval_quiz_stream_async(self):
        """Streaming version of generate_retrieval_quiz_async: yields the raw JSON
        text as it is generated. Consumers parse complete questions out of the
        growing text with parse_quiz_partial() and the finished text with
        _parse_and_validate_quiz(), so the first question can be shown long
        before the whole quiz is done."""
        if not self.context:
            return
        try:
            messages = [{"role": "user", "content": self._get_quiz_prompt()}]

            def factory(model):
                return self._make_async_openai_streaming_call(
                    messages=messages, model=model, temperature=0.3, max_tokens=15000, timeout=90,
                    reasoning_effort=QUIZ_REASONING_EFFORT, response_format={"type": "json_object"}
                )

            async for chunk in self._stream_with_fallback_for(QUIZ_MODEL, factory, "STREAM QUIZ", ""):
                yield chunk
        except Exception as e:
            logging.error(f"STREAM QUIZ: Critical error: {e}")

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
            # Classification is a short, cheap call: use the faster secondary model
            # first and keep the primary only as a fallback.
            for model in (MODEL_PRIMARY, MODEL_FALLBACK):
                try:
                    content = await self._make_async_openai_fallback_call(
                        messages, model=model, max_tokens=1000, timeout=30,
                        reasoning_effort=CLASSIFIER_REASONING_EFFORT
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

            for model in (EXTRACTION_MODEL, _fallback_for(EXTRACTION_MODEL)):
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

            for model in (EXTRACTION_MODEL, _fallback_for(EXTRACTION_MODEL)):
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
                logging.debug(f"ASYNC DEBUG: Trying async {CALC_MODEL} for calculation question generation...")
                result = await self._make_async_openai_fallback_call(
                    messages=messages,
                    model=CALC_MODEL,
                    max_tokens=8000,
                    timeout=120,
                    reasoning_effort="medium"
                )

                # Log raw API response
                logging.debug(f"RAW API RESPONSE:\n{result}")

                # No LaTeX formatting - let MathJax handle delimiters directly
                logging.info(f"Async calculation question generated successfully using {CALC_MODEL}")
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

        def factory(model):
            return self._make_async_openai_streaming_call(
                messages=messages, model=model, max_tokens=8000, timeout=120, reasoning_effort="medium"
            )

        # Same policy as the other features: retry the primary model, then fall
        # back to the secondary, and only then show the failure message.
        async for chunk in self._stream_with_fallback_for(CALC_MODEL, 
            factory, "CALC_QUESTION_STREAM", "I'm sorry, the AI service is taking too long to generate a calculation question right now. Please try again in a moment."
        ):
            yield chunk

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
                model=CALC_MODEL,
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

        def factory(model):
            return self._make_async_openai_streaming_call(
                messages=messages, model=model, max_tokens=8000, timeout=120, reasoning_effort="medium"
            )

        # Same policy as the other features: retry the primary model, then fall
        # back to the secondary, and only then show the failure message.
        async for chunk in self._stream_with_fallback_for(CALC_MODEL, 
            factory, "EXAM_WORKED_EXAMPLE_STREAM", "I'm sorry, the AI service is taking too long to generate a worked example right now. Please try again in a moment."
        ):
            yield chunk

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

        def factory(model):
            return self._make_async_openai_streaming_call(
                messages=messages, model=model, max_tokens=8000, timeout=120, reasoning_effort="medium"
            )

        # Same policy as the other features: retry the primary model, then fall
        # back to the secondary, and only then show the failure message.
        async for chunk in self._stream_with_fallback(
            factory, "CHECK_CALC_ANSWER_STREAM", f"**Feedback:** I received your answer: {user_answer}. However, I'm having trouble processing the evaluation right now. Please try again in a moment."
        ):
            yield chunk
