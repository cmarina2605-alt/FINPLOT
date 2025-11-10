# prompts.py — central place for FinPilot prompt text & builder
# -------------------------------------------------------------
# Purpose:
# - Keep the assistant's tone/rules in one file (easy to tweak).
# - Provide a helper (render_prompt) that formats:
#     SYSTEM + CONTEXT + QUESTION → final prompt string.
# - No external dependencies.

from typing import Optional

# High-level role + rules. Keep this concise for better model focus.
SYSTEM_PROMPT = """You are FinPilot, an internal assistant for financial advisors.
Answer ONLY with facts that appear in the provided CONTEXT. If the answer is not
explicitly supported by the CONTEXT, reply: "I don’t have that in the documents."

Style requirements:
- Be concise (about 5–8 lines).
- Use plain, non-technical language unless the user asks otherwise.
- Include inline citations like [Document.pdf p.X c.Y] near the facts they support.
- Never invent sources, numbers, or compliance claims.
- If policies conflict, say which document controls (by name + page).
"""

# Tiny note shown when we have to cut the context to fit the model window
TRUNCATION_NOTE = "[...context truncated for length…]"

# Default maximum characters for the context block (align with llm.py env default)
DEFAULT_MAX_CONTEXT_CHARS = 12_000


def render_prompt(
    question: str,
    context: str,
    max_context_chars: Optional[int] = None,
) -> str:
    """
    Formats the final prompt sent to the LLM:
      SYSTEM_PROMPT
      CONTEXT:
      <retrieved text with [Doc p.X c.Y] tags>

      QUESTION:
      <user question>

      FINAL ANSWER (with citations):
    """
    limit = max_context_chars or DEFAULT_MAX_CONTEXT_CHARS
    if len(context) > limit:
        context = context[:limit] + f"\n{TRUNCATION_NOTE}"

    return f"""{SYSTEM_PROMPT}

CONTEXT:
{context}

QUESTION:
{question}

FINAL ANSWER (with citations):
"""
