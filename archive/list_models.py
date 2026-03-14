#!/usr/bin/env python3
"""List available models for the google.genai client (uses GEMINI_API_KEY / GOOGLE_API_KEY from .env)."""

import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from google import genai

api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
if not api_key:
    print("Set GEMINI_API_KEY or GOOGLE_API_KEY in .env")
    exit(1)

client = genai.Client(api_key=api_key)
print("Available models (name):\n")
for m in client.models.list():
    name = getattr(m, "name", None) or str(m)
    print(f"  {name}")
print("\nDone.")
