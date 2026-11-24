"""Diabetes Risk Health-Check Agent.

An LLM agent that extracts health details from free text (English or Hinglish)
into a strict Pydantic schema, calls the trained scikit-learn models as tools,
and explains the result in plain language.
"""

__version__ = "2.0.0"
