
import os
import asyncio
from openai import OpenAI, AsyncOpenAI
import json
import logging

class TutorAI:

    def __init__(self):

        # Initialize OpenAI clients (primary and async)
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")

        self.openai_client = OpenAI(
            api_key=self.openai_api_key,
            max_retries=0
        )
        # Async client using default httpx transport (thread-safe across background threads)
        self.async_openai_client = AsyncOpenAI(
            api_key=self.openai_api_key,
            max_retries=0
        )

        # Async Gemini client via OpenAI-compatible endpoint (used for executive summary)
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
        if self.gemini_api_key:
            self.async_gemini_client = AsyncOpenAI(
                api_key=self.gemini_api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                max_retries=0
            )
        else:
            self.async_gemini_client = None

        self.context = None
        self.conversation_history = []

        # System prompt for the AI tutor
        self.system_prompt = r"""You are an intelligent and patient AI tutor. Your role is to help students learn and understand their lecture notes effectively. 
    
        Key behaviors:
        - Be encouraging and supportive
        - Break down complex concepts into digestible parts
        - Use examples and analogies to explain difficult topics
        - Ask follow-up questions to check for understanding
        - Provide practice questions and exercises when appropriate
        - Adapt your teaching style to the student's needs
        - Always base your responses on the provided lecture notes context
        - If asked about something not in the notes, acknowledge this and provide general guidance
        - Use emojis sparingly but appropriately to maintain engagement
    
        CRITICAL FORMATTING REQUIREMENTS FOR GENERAL CHAT:
        • **ABSOLUTELY NO MATHEMATICAL NOTATION:** You MUST NEVER include any mathematical equations, formulas, symbols, or LaTeX notation in general chat responses
          * NO dollar signs around variables: $x$, $\delta$, $P_t$, $\alpha$, etc.
          * NO backslash notation: \\(x\\), \\[equation\\], \\delta, \\alpha, etc.
          * NO mathematical symbols: √, ∑, ∫, ≤, ≥, ≠, π, etc.
          * If the lecture notes contain LaTeX variables like $P_t$ or $N_d2$, convert to plain text like "Pt" or "Nd2" (just remove dollar signs)
          * For Greek letters like $\delta$ or $\alpha$, write out the full word: "delta" or "alpha"
        • Use only plain text with markdown formatting (**bold**, *italic*, `code`, bullet points)
        • **HEADING FORMATTING**: Main headings in your responses must be in BLOCK CAPITALS and formatted with both bold and italic markdown: ***LIKE THIS***. 
        Sub-headings should be in BLOCK CAPITALS with bold markdown: **LIKE THIS**. 
        Keywords should be in italics *Like this*.
        • Use ONLY plain English words to describe all mathematical concepts
        • For example: Instead of writing \\(PV = \\frac{FV}{(1+r)^n}\\), write "Present value equals future value divided by one plus the interest rate raised to the power of n"
        • Instead of mathematical symbols, use words: "greater than", "less than", "equals", "multiplied by", "divided by", "squared", "cubed"
        • Instead of Greek letters in notation, write out the full word: "delta" not δ, "alpha" not α, "beta" not β
        • NEVER use HTML tags in your responses - only use markdown formatting
        • Always respond in plain text with markdown formatting - never return HTML, JSON, or other markup languages unless explicitly requested

        MATHEMATICAL FORMATTING (ONLY FOR SPECIALIZED FUNCTIONS):
        Note: The following LaTeX rules apply ONLY to specialized mathematical functions like calculation questions, NOT to general chat:
        - Format mathematical expressions using LaTeX syntax with proper delimiters (specialized functions only)
        - Use \\(...\\) for inline math and \\[...\\] for display math (specialized functions only)
        - Do NOT use $, $$, for LaTeX formatting (these delimiters have been disabled)
        - Do NOT use LaTeX spacing commands (\\;, \\!, \\,, \\:)
        - Do NOT use an overline or vinculum
        - Escape reserved characters and use proper math operator commands
    
        Remember: Your goal is to enhance learning, not just provide answers.
    
        ESSENTIAL: THIS IS A STUDY AND REVISION TOOL. NEVER ALLOW STUDENTS TO CHEAT BY PROVIDING EXTENSIVE ESSAY TYPE ANSWERS.
    
        ESSENTIAL: THIS TOOL IS ONLY TO BE USED FOR THE PURPOSES OF HELPING STUDENTS AT QUEEN'S UNIVERSITY BELFAST (QUB) TO STUDY AND REVISE FOR THEIR FINANCE COURSES. IT IS NOT TO BE USED FOR ANY OTHER PURPOSE.
    
        """

    # All AI calls go through _make_async_openai_fallback_call
    # Summary: Gemini Flash Lite primary, gpt-5.4-nano fallback, gpt-5.4-mini last resort
    # All other features: gpt-5.4-mini primary, gpt-5.4-nano fallback

    async def _make_async_openai_fallback_call(self, messages, model="gpt-5.4-nano", temperature=0.7, max_tokens=20000, response_format=None, timeout=50):

        if not self.async_openai_client:
            raise Exception("Async OpenAI fallback is not available - no API key configured.")
        
        try:
            # Reduced retries to fail faster
            max_retries = 1
            
            for attempt in range(max_retries + 1):
                try:
                    # Handle gpt-5.4-mini model which has different requirements
                    if model in ("gpt-5.4-mini", "gpt-5", "gpt-5-mini", "gpt-5.4-nano"):
                        # gpt-5.4-mini doesn't support system messages - combine all messages into user message
                        combined_content = ""
                        for message in messages:
                            if message["role"] == "system":
                                combined_content += f"System: {message['content']}\n\n"
                            else:
                                combined_content += f"{message['content']}\n\n"
                        
                        # Prepare arguments for gpt-5.4-mini (no system messages, use max_completion_tokens, no temperature)
                        api_args = {
                            "model": model,
                            "messages": [{"role": "user", "content": combined_content.strip()}],
                            "max_completion_tokens": max_tokens,
                        }
                        # gpt-5.4-mini doesn't support response_format parameter or temperature (only default value 1)
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

                    # Make the actual async API call with timeout
                    response = await asyncio.wait_for(
                        self.async_openai_client.chat.completions.create(**api_args),
                        timeout=timeout
                    )

                    # Extract the content from the response
                    content = response.choices[0].message.content

                    # Debug: Log response details
                    logging.debug(f"Async OpenAI response: finish_reason={response.choices[0].finish_reason}, content_length={len(content) if content else 0}")
                    
                    # Check for empty response
                    if not content or not content.strip():
                        finish_reason = response.choices[0].finish_reason
                        logging.error(f"Async OpenAI returned an empty response. Finish reason: {finish_reason}")
                        
                        # Handle different finish reasons
                        if finish_reason == "content_filter":
                            raise ValueError("Content was filtered by AI safety systems. Please try generating a different question.")
                        elif finish_reason == "length":
                            raise ValueError("Response was truncated due to length limits. Please try again.")
                        else:
                            raise ValueError("Async OpenAI service returned an empty response.")

                    return content

                except asyncio.TimeoutError:
                    logging.error(f"Async OpenAI API call timed out after {timeout} seconds (attempt {attempt + 1}/{max_retries + 1})")
                    if attempt < max_retries:
                        await asyncio.sleep(1)  # Brief delay before retry
                        continue
                    else:
                        raise Exception("The AI service is taking longer than expected to respond. Please try again.")

                except Exception as e:
                    # Log the detailed error for debugging purposes
                    logging.error(f"Async OpenAI API call failed (attempt {attempt + 1}/{max_retries + 1}). Error: {type(e).__name__} - {e}")
                    
                    # For SSL/connection errors, fail immediately to prevent worker kill
                    error_str = str(e).lower()
                    if any(pattern in error_str for pattern in ['ssl', 'sock', 'recv', 'read', 'connection', 'httpcore', 'systemexit']):
                        logging.error("Async SSL/connection error detected, failing immediately to prevent worker kill")
                        raise Exception("Connection to AI service failed. Please try again in a few moments.")
                    
                    # Check if this is a server error that we should retry
                    should_retry = any(pattern in error_str for pattern in ['500', '502', '503', '504', 'timeout'])
                    
                    if should_retry and attempt < max_retries:
                        logging.info(f"Retrying async API call in 1 second... (attempt {attempt + 1}/{max_retries + 1})")
                        await asyncio.sleep(1)
                        continue
                    
                    # If we've exhausted retries or it's not a retryable error, raise the exception
                    raise Exception("I'm having trouble connecting to the AI service right now. This is likely a temporary issue. Please try again in a few moments.")
                    
        except Exception as e:
            logging.error(f"Async OpenAI fallback call failed: {e}")
            raise e

    def set_context(self, pdf_content):

        self.context = pdf_content
        self.conversation_history = []  # Reset conversation when new context is set

    def clear_context(self):

        self.context = None
        self.conversation_history = []

    async def get_response_async(self, user_message):

        if not self.context:
            return "I need you to upload your lecture notes first before I can help you study! 📚"

        # Truncate context for optimal performance
        truncated_context = self.context[:80000] if len(self.context) > 80000 else self.context
        
        # Prepare the messages for the Gemini API
        system_content = f"{self.system_prompt}\n\nLecture Notes Context:\n{truncated_context}"
        
        # Add conversation history context if available
        if self.conversation_history:
            recent_history = self.conversation_history[-10:]  # Last 10 messages
            system_content += f"\n\nPrevious Conversation Context:\nThe following are the most recent messages from our ongoing conversation. Please build on this context when responding to the user's latest query. Use this history to maintain continuity and reference previous topics discussed:\n"
            for i, msg in enumerate(recent_history):
                role_label = "Student" if msg["role"] == "user" else "AI Tutor"
                system_content += f"{role_label}: {msg['content']}\n"
            system_content += "\nPlease use this conversation history to provide a contextually relevant response to the user's new message below."
        
        messages = [
            {
                "role": "system",
                "content": system_content
            }
        ]

        # Add current user message
        messages.append({"role": "user", "content": user_message})

        try:
            # Use OpenAI gpt-5.4-mini as primary for general chat
            ai_response = await self._make_async_openai_fallback_call(
                messages=messages,
                model="gpt-5.4-mini",
                temperature=0.7,
                max_tokens=15000,
                timeout=60
            )
            
            # Update conversation history
            self.conversation_history.append({"role": "user", "content": user_message})
            self.conversation_history.append({"role": "assistant", "content": ai_response})
            
            return ai_response
            
        except Exception as openai_error:
            # Fallback to gpt-5.4-nano if gpt-5.4-mini fails
            try:
                ai_response = await self._make_async_openai_fallback_call(
                    messages=messages,
                    model="gpt-5.4-nano",
                    temperature=0.7,
                    max_tokens=15000,
                    timeout=60
                )
                
                # Update conversation history
                self.conversation_history.append({"role": "user", "content": user_message})
                self.conversation_history.append({"role": "assistant", "content": ai_response})
                
                return ai_response
                
            except Exception as nano_error:
                logging.error(f"Both gpt-5.4-mini and gpt-5.4-nano failed for general chat: {openai_error} | {nano_error}")
                return "I'm having trouble connecting to the AI service right now. This is likely a temporary issue. Please try again in a few moments."

    

    async def generate_cheat_sheet_async(self):

        if not self.context:
            return "No lecture notes available to create sheet from."

        try:
            prompt = """

Create a comprehensive study aid from the lecture notes I provide. Your output should begin with a concise overview, followed by a detailed bullet-point revision sheet.

### CRITICAL FORMATTING REQUIREMENTS - FOLLOW EXACTLY
- **ONLY USE HYPHENS FOR BULLETS:** You MUST use only hyphens (`-`) for ALL bullet points. Do NOT use asterisks (*), bullet symbols (•), or any other characters. EVERY bullet point must start with a hyphen.
- **Bullet Point Format:** Each bullet point must follow this exact format: `- *Concept:* Brief explanation`
- **New Lines:** Ensure every bullet point is on a new line with proper spacing.
- **Logical Structure:** Organize information logically. Use indentation for sub-bullets to create a clear hierarchy.
- **Readability:** The final output must be scan-friendly and easy to reference quickly.
- **ABSOLUTELY NO MATHEMATICAL NOTATION:** You MUST NOT include ANY mathematical equations, formulas, symbols, or LaTeX notation of any kind. This is CRITICAL - do NOT use:
  * Dollar signs around variables: NO $x$, $\delta$, $P_t$, $\alpha$, etc.
  * Backslash notation: NO \(x\), \[equation\], \delta, \alpha, etc.
  * Mathematical symbols: NO √, ∑, ∫, ≤, ≥, ≠, π, etc.
  * If the lecture notes contain LaTeX like $P_t$ or $N_d2$, convert to plain text like "Pt" or "Nd2" (just remove the dollar signs)
  * For Greek letters like $\delta$ or $\alpha$, write out the full word: "delta" or "alpha"
  * Use ONLY plain text to describe all mathematical concepts

---

### REQUIRED OUTPUT STRUCTURE

You MUST use the following structure and formatting precisely.

***OVERVIEW***

Summarise the main subject/topic of the lecture. Write it in a professional, academic tone suitable for quick review before an exam.

***KEY CONCEPTS***

List the most important concepts with brief definitions, grouped into logical categories. Each concept must be on its own line. Follow this exact template:

**1. [First Category Name]**

- *[Concept]:* Brief explanation of the concept.
- *[Concept]:* Brief explanation of the concept.
- *[Concept]:* Brief explanation of the concept.

**2. [Second Category Name]**

- *[Concept]:* Brief explanation of the concept.
- *[Concept]:* Brief explanation of the concept.

**3. [Third Category Name]**

- *[Concept]:* Brief explanation of the concept.
- *[Concept]:* Brief explanation of the concept.

End your response with: "Would you like to explore any of these topics in more detail?"

            
            """

            # Truncate context to 80,000 characters for async API calls
            truncated_context = self.context[:80000] if len(self.context) > 80000 else self.context
            
            messages = [
                {
                    "role": "system",
                    "content": f"You are an expert at creating study aids and revision sheets from academic content.\n\nLecture Notes:\n{truncated_context}"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]

            logging.info(f"ASYNC SUMMARY: Context length: {len(truncated_context)} characters")

            # Try Gemini Flash Lite as primary
            if self.async_gemini_client:
                try:
                    logging.info("ASYNC SUMMARY: Trying gemini-flash-lite-latest for executive summary generation...")
                    gemini_response = await asyncio.wait_for(
                        self.async_gemini_client.chat.completions.create(
                            model="gemini-flash-lite-latest",
                            messages=messages,
                            temperature=0.2,
                            max_tokens=15000,
                        ),
                        timeout=60
                    )
                    result = gemini_response.choices[0].message.content
                    if result and result.strip():
                        logging.info("ASYNC SUMMARY: gemini-flash-lite-latest succeeded")
                        return result
                    raise ValueError("Gemini returned an empty response")
                except Exception as gemini_error:
                    logging.error(f"ASYNC SUMMARY: gemini-flash-lite-latest failed: {gemini_error}")

            # Fallback to gpt-5.4-nano
            try:
                logging.info("ASYNC SUMMARY: Trying gpt-5.4-nano fallback...")
                result = await self._make_async_openai_fallback_call(
                    messages=messages,
                    model="gpt-5.4-nano",
                    temperature=0.2,
                    max_tokens=15000,
                    timeout=60
                )
                logging.info("ASYNC SUMMARY: gpt-5.4-nano fallback succeeded")
                return result
            except Exception as nano_error:
                logging.error(f"ASYNC SUMMARY: gpt-5.4-nano failed: {nano_error}")

            # Last resort: gpt-5.4-mini
            try:
                logging.info("ASYNC SUMMARY: Trying gpt-5.4-mini last resort...")
                result = await self._make_async_openai_fallback_call(
                    messages=messages,
                    model="gpt-5.4-mini",
                    temperature=0.7,
                    max_tokens=15000,
                    timeout=60
                )
                logging.info("ASYNC SUMMARY: gpt-5.4-mini last resort succeeded")
                return result
            except Exception as mini_error:
                logging.error(f"ASYNC SUMMARY: All models failed: {mini_error}")
                return "I'm having trouble generating a summary right now. The document appears to be loaded successfully, but there may be a temporary issue with the AI service. Please try again in a moment or use the chat to ask specific questions about your document."

        except Exception as e:
            logging.error(f"ASYNC SUMMARY: Critical error in async summary generation: {e}")
            return f"Critical error in summary generation: {str(e)}"

    async def generate_essay_question_async(self):

        if not self.context:
            return "No lecture notes available to create essay question from."

        try:
            prompt = """

            CRITICAL FORMATTING REQUIREMENTS:
            - Use markdown formatting for emphasis: **bold text**, *italic text*, `code text`
            - **ABSOLUTELY NO MATHEMATICAL NOTATION:** Do NOT use LaTeX formatting, mathematical symbols, or any notation:
              * NO dollar signs: $x$, $\delta$, $P_t$, etc.
              * NO backslash notation: \(x\), \[equation\], etc.
              * NO mathematical symbols: √, ∑, ∫, ≤, ≥, ≠, π, etc.
              * If the lecture notes contain LaTeX variables like $P_t$ or $N_d2$, convert to plain text like "Pt" or "Nd2" (just remove dollar signs)
              * For Greek letters like $\delta$ or $\alpha$, write out the full word: "delta" or "alpha"
            - NEVER use HTML tags - only use markdown formatting
            - ONLY USE HYPHENS FOR BULLETS (-) - never use asterisks (*) or dots (•)
            - Each bullet point must be on its own line with consistent hyphen formatting
            - Use ONLY plain English words to describe ALL mathematical concepts
            - Always respond in plain text with markdown formatting only


            TASK:

            **ESSAY QUESTION:**

            Create ONE substantial essay question, with several suggested sub-questions, that:
            • Requires integration of multiple concepts from the lecture notes, and 
            • Asks for the citation of additional reading of other academic literature, and 
            • Asks for commentary on real-world applications
            • Asks for analysis, evaluation, or application (not just description)
            • Is answerable in 500-750 words

            **🎯 ASSESSMENT CRITERIA (based on QUB Conceptual Equivalents Scale):**

            **FIRST CLASS (70-100%):**
            
            **Exceptional (90-100%):** 
            Exceptional and exemplary work showing:
            A very high level of critical analysis.
            A very high level of insight in the conclusions drawn.
            An in-depth knowledge and understanding across a wide range of relevant areas including areas at the forefront of the discipline.
            Very thorough coverage of the topic.
            Confidence in the appropriate use of learning resources to support arguments made.
            
            **Definite First (80-89%):** 
            Excellent and outstanding answer showing:
            Considerable independence of thought and critical judgement with sustained critical analysis.
            A well-developed ability to analyse concepts and ideas at an abstract level.
            A thorough understanding of all the main issues involved and their relevance.
            A substantial degree of originality.
            Substantial evidence of wide, relevant and critical use of learning resources.
            Good understanding of complex and problematic areas of the discipline.
            
            **Low First (70-79%):**
            
            Excellent answer showing:
            A good level of independence of thought and critical judgement and a level of critical analysis.
            A developed ability to analyse concepts and ideas.
            An understanding of all the main issues involved and their relevance.
            A degree of originality.
            Evidence of wide, relevant and critical use of learning resources.
            Some understanding of complex and problematic areas of the discipline.

            **UPPER SECOND CLASS (60-69%):**
            
            **High 2:1 (65-69%):**
            Good performance showing:
            Some independence of thought and critical judgement.
            Some ability to analyse concepts and ideas.
            An understanding of the main issues involved and their relevance.
            Appropriate use of learning resources.
            Clear understanding of a reasonable range of literature or source materials.
            
            **Low 2:1 (60-64%):**
            
            Good performance showing:
            Clear understanding of a reasonable range of literature or source materials.
            Some knowledge and understanding of the central issues and themes.
            Appropriate use of learning resources.
            Some level of critical analysis and evaluation.

            **LOWER SECOND CLASS (50-59%):**
            
            Adequate answer showing:
            Some knowledge and understanding of the central issues and themes.
            Limited critical analysis and evaluation.
            Limited literature covered.
            Average understanding of materials.
            Limited independence of thought.

            **THIRD CLASS (40-49%):**
            
            Weak answer showing:
            Demonstration of basic knowledge.
            Limited understanding of the topic area.
            Some irrelevance of content.
            Uncritical use of sources.
            Little indication of independent learning.

---

### REQUIRED OUTPUT STRUCTURE

You MUST use the following structure and formatting precisely.

```markdown

***ESSAY QUESTION***


**MAIN QUESTION**

[The primary essay prompt]

**SUB-QUESTIONS**

[2-3 guiding questions to help structure the response]

**KEY CONCEPTS TO ADDRESS**

[Explain how 4-5 specific concepts could be included]

**ASSESSMENT CRITERIA**

[Explain the QUB Conceptual Equivalents Scale in detail]

**WRITING TIPS**

[Provide 3-5 concise tips for writing the essay (do not mention formatting of mathematical notation)]


            """

            # Truncate context to 80,000 characters for async API calls
            truncated_context = self.context[:80000] if len(self.context) > 80000 else self.context
            
            # Try async Gemini as primary
            try:
                logging.info("ASYNC ESSAY: Trying gpt-5.4-mini for essay question generation...")
                logging.info(f"ASYNC ESSAY: Context length: {len(truncated_context)} characters")
                
                result = await self._make_async_openai_fallback_call(
                    messages=[
                        {
                            "role": "system",
                            "content": f"You are an expert at creating analytical essay questions from academic content.\n\nLecture Notes:\n{truncated_context}"
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    model="gpt-5.4-mini",
                    temperature=0.4,
                    max_tokens=15000,
                    timeout=60
                )
                
                logging.info("ASYNC ESSAY: gpt-5.4-mini succeeded")
                return result
                
            except Exception as openai_error:
                logging.error(f"ASYNC ESSAY: gpt-5.4-mini failed: {openai_error}")
                # Fallback to gpt-5.4-nano
                try:
                    logging.info("ASYNC ESSAY: Trying gpt-5.4-nano fallback...")
                    fallback_result = await self._make_async_openai_fallback_call(
                        messages=[
                            {
                                "role": "system",
                                "content": f"You are an expert at creating analytical essay questions from academic content.\n\nLecture Notes:\n{truncated_context}"
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        model="gpt-5.4-nano",
                        temperature=0.4,
                        max_tokens=15000,
                        timeout=60
                    )
                    
                    logging.info("ASYNC ESSAY: gpt-5.4-nano fallback succeeded")
                    return fallback_result
                    
                except Exception as nano_error:
                    logging.error(f"ASYNC ESSAY: Both gpt-5.4-mini and gpt-5.4-nano failed: {nano_error}")
                    return "I'm having trouble generating an essay question right now. The document appears to be loaded successfully, but there may be a temporary issue with the AI service. Please try again in a moment or use the chat to ask specific questions about your document."

        except Exception as e:
            logging.error(f"ASYNC ESSAY: Critical error in async essay generation: {e}")
            return f"Critical error in essay generation: {str(e)}"

    async def explain_key_concepts_async(self):

        if not self.context:
            return "No lecture notes available to explain key concepts from."

        try:
            prompt = r"""

Identify and briefly explain exactly 5 key concepts from these lecture notes. Present them in a clear, accessible way that helps students understand complex ideas without being condescending. 

CRITICAL FORMATTING REQUIREMENTS:
- **ABSOLUTELY NO MATHEMATICAL NOTATION:** Do NOT use LaTeX formatting, mathematical equations, symbols, or any notation anywhere in your response
  * NO dollar signs around variables: $x$, $\delta$, $P_t$, etc.
  * NO backslash notation: \(x\), \[equation\], etc.
  * NO mathematical symbols: √, ∑, ∫, ≤, ≥, ≠, π, etc.
  * If the lecture notes contain LaTeX variables like $P_t$ or $N_d2$, convert to plain text like "Pt" or "Nd2" (just remove dollar signs)
  * For Greek letters like $\delta$ or $\alpha$, write out the full word: "delta" or "alpha"
- Use only plain text with markdown formatting (bold, italic, bullet points)
- ONLY USE HYPHENS FOR BULLETS (-) - never use asterisks (*) or dots (•)
- Each bullet point must be on its own line with consistent hyphen formatting
- Describe ALL mathematical concepts using ONLY plain English words

Use the following structure. Each bullet point MUST be on its own line.

---

### REQUIRED OUTPUT STRUCTURE

You MUST use the following structure and formatting precisely.

```markdown

***KEY CONCEPTS EXPLAINED:***

For each major concept you identify:

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

End the overall response with: "Would you like to explore any of these topics in more detail?"
            
            """

            # Truncate context to 80,000 characters for async API calls
            truncated_context = self.context[:80000] if len(self.context) > 80000 else self.context
            
            # Try async Gemini as primary
            try:
                logging.info("ASYNC KEY CONCEPTS: Trying gpt-5.4-mini for key concepts explanation...")
                logging.info(f"ASYNC KEY CONCEPTS: Context length: {len(truncated_context)} characters")
                
                result = await self._make_async_openai_fallback_call(
                    messages=[
                        {
                            "role": "system",
                            "content": f"\n\nLecture Notes:\n{truncated_context}"
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    model="gpt-5.4-mini",
                    temperature=0.4,
                    max_tokens=15000,
                    timeout=60
                )
                
                logging.info("ASYNC KEY CONCEPTS: gpt-5.4-mini succeeded")
                # Return raw result without LaTeX preprocessing
                return result
                
            except Exception as openai_error:
                logging.error(f"ASYNC KEY CONCEPTS: gpt-5.4-mini failed: {openai_error}")
                # Fallback to gpt-5.4-nano
                try:
                    logging.info("ASYNC KEY CONCEPTS: Trying gpt-5.4-nano fallback...")
                    fallback_result = await self._make_async_openai_fallback_call(
                        messages=[
                            {
                                "role": "system",
                                "content": f"\n\nLecture Notes:\n{truncated_context}"
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        model="gpt-5.4-nano",
                        temperature=0.4,
                        max_tokens=15000,
                        timeout=60
                    )
                    
                    logging.info("ASYNC KEY CONCEPTS: gpt-5.4-nano fallback succeeded")
                    # Return raw result without LaTeX preprocessing
                    return fallback_result
                    
                except Exception as nano_error:
                    logging.error(f"ASYNC KEY CONCEPTS: Both gpt-5.4-mini and gpt-5.4-nano failed: {nano_error}")
                    return "I'm having trouble explaining the key concepts right now. The document appears to be loaded successfully, but there may be a temporary issue with the AI service. Please try again in a moment or use the chat to ask specific questions about your document."

        except Exception as e:
            logging.error(f"ASYNC KEY CONCEPTS: Critical error in async key concepts explanation: {e}")
            return f"Critical error in key concepts explanation: {str(e)}"

    async def generate_retrieval_quiz_async(self):

        if not self.context:
            return []

        try:
            # Truncate context to 80,000 characters for async API calls
            context_truncated = self.context[:80000] if len(self.context) > 80000 else self.context
            
            prompt = f"""Based on these lecture notes, create exactly 15 simple multiple choice questions about the key concepts. Do NOT include any mathematical equations or formulas.

{context_truncated}

CRITICAL INSTRUCTIONS:
1. Your response must start immediately with {{ and end with }} - NO other characters
2. Do NOT include any text, explanations, or formatting before or after the JSON
3. Do NOT include markdown code blocks, backticks, or any other formatting
4. Do NOT include any HTML tags or special characters outside the JSON
5. Use this EXACT JSON structure and format:

{{
    "questions": [
        {{
            "question": "What is the main topic of this lecture?",
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

Make questions simple and focused on basic concepts. Keep explanations short and informative. 

CRITICAL FORMATTING RULES:
- Do NOT start explanations with ANY confirmation words like 'Correct!', 'Right!', 'Yes!', 'That's correct!', 'Exactly!', or any similar phrases. Start explanations directly with the educational content.
- **ABSOLUTELY NO MATHEMATICAL NOTATION:** Do NOT use LaTeX formatting, mathematical symbols, or any notation in questions, options, or explanations:
  * NO dollar signs: $x$, $\delta$, $P_t$, etc.
  * NO backslash notation: \(x\), \[equation\], etc.
  * NO mathematical symbols: √, ∑, ∫, ≤, ≥, ≠, π, etc.
  * If the lecture notes contain LaTeX variables like $P_t$ or $N_d2$, convert to plain text like "Pt" or "Nd2" (just remove dollar signs)
  * For Greek letters like $\delta$ or $\alpha$, write out the full word: "delta" or "alpha"
- Use ONLY plain English words to describe all mathematical concepts
- Use only plain text - no mathematical notation or formulas

RESPONSE FORMAT: Start your response with {{ immediately - no whitespace, no text, no code blocks."""

            # Try gpt-5.4-mini as primary
            try:
                logging.info("ASYNC QUIZ: Trying gpt-5.4-mini for retrieval quiz generation...")
                logging.info(f"ASYNC QUIZ: Context length: {len(context_truncated)} characters")
                
                result = await self._make_async_openai_fallback_call(
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    model="gpt-5.4-mini",
                    response_format={"type": "json_object"},
                    temperature=0.3,
                    max_tokens=15000,
                    timeout=60
                )
                
                logging.info("ASYNC QUIZ: gpt-5.4-mini succeeded")
                
                # Parse and validate the JSON response
                import json
                
                # Clean the response - remove any leading/trailing whitespace and common prefixes
                cleaned_content = result.strip()
                
                # Remove common markdown code block patterns
                if cleaned_content.startswith('```json'):
                    cleaned_content = cleaned_content[7:]  # Remove ```json
                if cleaned_content.startswith('```'):
                    cleaned_content = cleaned_content[3:]  # Remove ```
                if cleaned_content.endswith('```'):
                    cleaned_content = cleaned_content[:-3]  # Remove trailing ```
                
                # Remove any leading text before the first {
                first_brace = cleaned_content.find('{')
                if first_brace > 0:
                    cleaned_content = cleaned_content[first_brace:]
                
                # Remove any trailing text after the last }
                last_brace = cleaned_content.rfind('}')
                if last_brace != -1 and last_brace < len(cleaned_content) - 1:
                    cleaned_content = cleaned_content[:last_brace + 1]
                
                # Parse JSON
                parsed_result = json.loads(cleaned_content)
                
                if "questions" in parsed_result:
                    questions = parsed_result["questions"][:15]  # Limit to 15 questions
                    
                    # Validate questions
                    valid_questions = []
                    for i, q in enumerate(questions):
                        # Check required fields
                        if not all(key in q for key in ["question", "options", "correct_answer", "explanation"]):
                            logging.warning(f"Async quiz question {i+1} missing required fields")
                            continue
                        
                        # Validate options structure
                        if not isinstance(q["options"], list) or len(q["options"]) != 4:
                            logging.warning(f"Async quiz question {i+1} options invalid")
                            continue
                        
                        # Validate correct_answer matches one of the options
                        correct_answer = q["correct_answer"].strip()
                        options = [opt.strip() for opt in q["options"]]
                        
                        if correct_answer not in options:
                            # Try to find a matching option
                            matched = False
                            for option in options:
                                if option.replace('Option A:', '').replace('Option B:', '').replace('Option C:', '').replace('Option D:', '').strip() == correct_answer.replace('Option A:', '').replace('Option B:', '').replace('Option C:', '').replace('Option D:', '').strip():
                                    q["correct_answer"] = option  # Fix the correct answer
                                    matched = True
                                    break
                            if not matched:
                                logging.warning(f"Async quiz question {i+1} correct_answer mismatch")
                                continue
                        
                        # Validate no empty fields
                        if not q["question"].strip() or not q["explanation"].strip():
                            logging.warning(f"Async quiz question {i+1} has empty fields")
                            continue
                        
                        valid_questions.append(q)
                    
                    logging.info(f"ASYNC QUIZ: Generated {len(valid_questions)} valid questions")
                    return valid_questions
                
                return []
                
            except Exception as openai_error:
                logging.error(f"ASYNC QUIZ: gpt-5.4-mini failed: {openai_error}")
                # Fallback to gpt-5.4-nano
                try:
                    logging.info("ASYNC QUIZ: Trying gpt-5.4-nano fallback...")
                    fallback_result = await self._make_async_openai_fallback_call(
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        model="gpt-5.4-nano",
                        response_format={"type": "json_object"},
                        temperature=0.3,
                        max_tokens=15000,
                        timeout=60
                    )
                    
                    logging.info("ASYNC QUIZ: Async OpenAI fallback succeeded")
                    
                    # Parse and validate the fallback JSON response
                    import json
                    
                    # Clean the response
                    cleaned_fallback = fallback_result.strip()
                    
                    # Remove common markdown code block patterns
                    if cleaned_fallback.startswith('```json'):
                        cleaned_fallback = cleaned_fallback[7:]
                    if cleaned_fallback.startswith('```'):
                        cleaned_fallback = cleaned_fallback[3:]
                    if cleaned_fallback.endswith('```'):
                        cleaned_fallback = cleaned_fallback[:-3]
                    
                    # Parse JSON
                    fallback_parsed = json.loads(cleaned_fallback)
                    
                    if "questions" in fallback_parsed:
                        questions = fallback_parsed["questions"][:15]
                        
                        # Validate questions (same validation as primary)
                        valid_questions = []
                        for i, q in enumerate(questions):
                            if all(key in q for key in ["question", "options", "correct_answer", "explanation"]):
                                if isinstance(q["options"], list) and len(q["options"]) == 4:
                                    # Basic validation - ensure correct_answer matches an option
                                    correct_answer = q["correct_answer"].strip()
                                    options = [opt.strip() for opt in q["options"]]
                                    
                                    if correct_answer in options:
                                        valid_questions.append(q)
                        
                        logging.info(f"ASYNC QUIZ: Fallback generated {len(valid_questions)} valid questions")
                        return valid_questions
                    
                    return []
                    
                except Exception as openai_error:
                    logging.error(f"ASYNC QUIZ: Both async methods failed: {openai_error}")
                    return []

        except Exception as e:
            logging.error(f"ASYNC QUIZ: Critical error in async quiz generation: {e}")
            return []

    # Legacy sync method removed - all quiz operations now use async polling

    async def generate_calculation_question_async(self, used_questions=None):

        if not self.context:
            return "No document context available. Please upload a document first."

        try:
            # Truncate context for gpt-5.4-mini with increased limit for better mathematical context
            context_truncated = self.context[:80000] if len(self.context) > 80000 else self.context

            # Build used questions context
            used_questions_context = ""
            if used_questions and len(used_questions) > 0:
                used_questions_context = f"""
PREVIOUSLY USED QUESTIONS (DO NOT REPEAT):
{chr(10).join([f"{i+1}. {q[:150]}..." for i, q in enumerate(used_questions)])}

IMPORTANT: Generate a NEW question that uses a DIFFERENT equation/formula from the ones already used above. Ensure variety in topics and mathematical concepts.
"""

            prompt = f"""Based on the following lecture notes, generate ONE calculation question that follows this specific 6-part layout pattern:

            LECTURE NOTES:
            {context_truncated}

            {used_questions_context}

            REQUIRED LAYOUT PATTERN:
            1) Choose ONE equation from the lecture notes and display it using LaTeX formatting. 
            2) Move sequentially through the major equations in the lecture notes. If there are no previously used questions, start with the first equation. If there are previously used questions, move to the next equation.
            3) Explain the variable definitions clearly
            4) Provide an explanation of what the equation means and its purpose
            5) Show a worked example using specific input values with step-by-step LaTeX calculations
            6) Set a challenge for the user using different input values
            7) Ask the user to input their answer in the chat

            CONTENT REQUIREMENTS:
            - Use ONLY mathematical formulas/equations that appear DIRECTLY in the lecture notes above
            - For worked examples, you may use values from the lecture notes if available
            - For challenge problems, you MUST create NEW and DIFFERENT numerical values
            - Show complete step-by-step calculations in LaTeX format
            - End by asking the user to type their numerical answer in the chat

            FORMATTING REQUIREMENTS:
            - Use standard LaTeX: \\[ equation \\] for display math, \\( variable \\) for inline math
            - Use standard math operators: \\times, \\div, \\cdot, \\frac{{numerator}}{{denominator}}
            - For subscripts: Always use underscore with braces \\mu_{{12}} (proper braces required)
            - For superscripts: Always use caret with braces \\sigma^{{2}} (proper braces required)  
            - For combined: \\hat{{\\mu}}_{{12}} or \\sigma_{{1}}^{{2}} (always use proper braces)
            - CRITICAL: Every subscript and superscript MUST have proper braces like _{{value}} and ^{{value}}
            - For the worked examples, you MUST use \\begin{{align*}} with proper alignment for each step so that the calculations are clear and easy to follow.
            - For matrices: \\begin{{bmatrix}} a & b \\\\ c & d \\end{{bmatrix}}. 
            - CRITICAL: NEVER use \\\\[6pt] or \\\\[4pt] or \\\\[3pt] or any other \\\\[pt], just use backslashes \\\\ for new lines and rows. For example, do NOT use \\begin{{bmatrix}} 14 & -5 \\\\ -8 & 10 \\end{{bmatrix}} \\\\[6pt], instead use \\begin{{bmatrix}} 14 & -5 \\\\ -8 & 10 \\end
            - For worked examples: Use **Step N:** Description followed by calculation
            - Use **bold** for section headers and step descriptions




            EXAMPLE FORMAT:

            ***CALCULATION QUESTION***

            **EQUATION**
            One of the equations used in this topic is:

            \\[ LaTeX equation here \\]

            The variables in this equation are: \\(x\\): Variable description; \\(y\\): Another variable description; and \\(z\\): Another variable description


            **EXPLANATION**
            Explanation of the equation's purpose and application.


            **WORKED EXAMPLE**
            Given values from lecture notes: \\(x = 10\\); \\(y = 5\\); and \\(z = 2\\)

            **Step 1:** Calculate the sum
            \\begin{{align*}}
            x + y + z &= 10 + 5 + 2\\\\
            &= 17
            \\end{{align*}}

            **Step 2:** Multiply by 2
            \\begin{{align*}}
            \\text{{result}} \\times 2 &= 17 \\times 2 \\\\
            &= 34
            \\end{{align*}}

            **Final Result:** 
            \\begin{{align*}}
            \\text{{Answer}} &= 34
            \\end{{align*}}

            **CHALLENGE**
            Calculate the result when: \\(x = 12\\); \\(y = 8\\); and \\(z = 10\\)

            Please type your numerical answer in the chat below.

            RESPONSE FORMAT: Provide the formatted text directly - no JSON, no code blocks, just the formatted calculation question."""

            # Try gpt-5.4-mini as primary model for calculation questions (with async)
            messages = [{"role": "user", "content": prompt}]
            
            try:
                logging.info("ASYNC DEBUG: Trying async gpt-5.4-mini for calculation question generation...")
                logging.info(f"ASYNC DEBUG: async_openai_client is available: {self.async_openai_client is not None}")
                result = await self._make_async_openai_fallback_call(
                    messages=messages,
                    model="gpt-5.4-mini",
                    max_tokens=5000,  # Increased to allow for more detailed mathematical explanations
                    timeout=90  # Extended to 90-second timeout for maximum reliability
                )
                
                # Log raw API response
                logging.info(f"RAW API RESPONSE:\n{result}")
                
                # No LaTeX formatting - let MathJax handle delimiters directly
                logging.info("CALC_QUESTION: Skipping LaTeX formatting, returning raw result")
                
                logging.info("Async calculation question generated successfully using gpt-5.4-mini")
                return result
                
            except Exception as e:
                logging.error(f"Async generation failed after all attempts: {e}")
                # Instead of falling back to sync (which blocks the event loop), return a clean error message
                return "I'm sorry, the AI service is taking too long to generate a calculation question right now. Please try again in a moment."

        except Exception as e:
            logging.error(f"Async calculation question generation failed: {e}")
            return "Sorry, I couldn't generate calculation questions at this time. Please try again later."

    async def check_calculation_answer_async(self, challenge_question, user_answer):

        if not self.context:
            return "No document context available."

        # Truncate context for optimal performance
        truncated_context = self.context[:80000] if len(self.context) > 80000 else self.context
        
        prompt = f"""You are evaluating a student's answer to a calculation question.

LECTURE NOTES CONTEXT:
{truncated_context}

ORIGINAL CHALLENGE QUESTION:
{challenge_question}

STUDENT'S ANSWER: {user_answer}

EVALUATION TASKS:
1. Calculate the correct answer using the provided challenge values
2. Provide the correct step-by-step solution
3. Determine if the student's answer is correct (accept reasonable rounding)
4. Give appropriate feedback
5. Ask if they want another calculation question

RESPONSE FORMAT:
**Step-by-Step Solution:**
**Step 1:** [Description of first step]
\\begin{{align*}}
[equation 1] &= [step 1] \\\\
&= [step 2] \\\\
&= [final result]
\\end{{align*}}
**Step 2:** [Description of second step]  
\\begin{{align*}}
[equation 2] &= [step 1] \\\\
&= [step 2] \\\\
&= [final result]
\\end{{align*}}
[Continue for all steps...]

**Final Answer:** [Correct numerical answer]

**Feedback:** [Comment on the student's answer]

**To move on to the next set of equations from the lecture notes, just click the Next Question button.**

CRITICAL FORMATTING REQUIREMENTS:
- ALWAYS use \\begin{{align*}} environment for ALL step-by-step calculations
- Use standard LaTeX: \\( variable \\) for inline math in text descriptions
- Use standard math operators: \\times, \\div, \\cdot, \\frac{{numerator}}{{denominator}}
- For subscripts: Always use underscore with braces \\mu_{{12}} (proper braces required)
- For superscripts: Always use caret with braces \\sigma^{{2}} (proper braces required)  
- For combined: \\hat{{\\mu}}_{{12}} or \\sigma_{{1}}^{{2}} (always use proper braces)
- CRITICAL: Every subscript and superscript MUST have proper braces like _{{value}} and ^{{value}}
- For matrices: \\begin{{bmatrix}} a & b \\\\ c & d \\end{{bmatrix}}
- CRITICAL: NEVER use \\\\[6pt] or \\\\[4pt], just use backslashes \\\\ for new lines and rows. For example, do NOT use \\begin{{bmatrix}} 14 & -5 \\\\ -8 & 10 \\end{{bmatrix}} \\\\[6pt], instead use \\begin{{bmatrix}} 14 & -5 \\\\ -8 & 10 \\end
- Use **bold** for section headers and step descriptions
- Each step must be in its own align* environment for proper formatting

IMPORTANT MULTI-LINE CALCULATION FORMATTING:
INCORRECT:
\\[
K\\,e^{{-rT}}
= 52 \\times e^{{-0.05 \\times 1}}
= 52 \\times e^{{-0.05}}
\\approx 52 \\times 0.951229
\\approx 49.4629
\\]

CORRECT:
\\begin{{align*}}
    K e^{{-rT}} &= 52 \\times e^{{-0.05 \\times 1}} \\\\
              &= 52 \\times e^{{-0.05}} \\\\
              &\\approx 52 \\times 0.951229 \\\\
              &\\approx 49.4629
\\end{{align*}}

"""

        # Try gpt-5.4-mini as primary model for calculation answer evaluation (using same approach as question generation)
        messages = [{"role": "user", "content": prompt}]
        
        try:
            logging.info("CHECK_CALC_ANSWER: Trying async gpt-5.4-mini for calculation answer evaluation...")
            response = await self._make_async_openai_fallback_call(
                messages=messages,
                model="gpt-5.4-mini",
                max_tokens=5000,
                timeout=90
            )
            
            logging.info("CHECK_CALC_ANSWER: RAW API RESPONSE from gpt-5.4-mini:")
            logging.info(f"---START RAW API RESPONSE---")
            logging.info(response)
            logging.info(f"---END RAW API RESPONSE---")
            
            # No LaTeX formatting - let MathJax handle delimiters directly
            logging.info("CHECK_CALC_ANSWER: gpt-5.4-mini success - returning raw response")
            
            return response
            
        except Exception as o4_error:
            logging.error(f"gpt-5.4-mini failed for calculation answer check: {o4_error}")
            # Fallback to gpt-5.4-nano
            try:
                logging.info("CHECK_CALC_ANSWER: Falling back to gpt-5.4-nano...")
                nano_response = await self._make_async_openai_fallback_call(
                    messages=messages,
                    model="gpt-5.4-nano",
                    max_tokens=5000,
                    timeout=45
                )
                
                # No LaTeX formatting - let MathJax handle delimiters directly
                logging.debug("CHECK_CALC_ANSWER: gpt-5.4-nano fallback success")
                
                return nano_response
                
            except Exception as nano_error:
                logging.error(f"Both gpt-5.4-mini and gpt-5.4-nano failed for calculation answer check: {o4_error} | {nano_error}")
                return f"**Feedback:** I received your answer: {user_answer}. However, I'm having trouble processing calculation evaluations right now. Please try again in a moment, or click the Calculation questions button to get a new question."

    