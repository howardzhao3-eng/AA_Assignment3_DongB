"""
DeepSeek API wrapper for experiment use.

Handles:
  - Authentication via environment variable (.env)
  - Rate limiting with exponential backoff
  - Response validation and error handling
  - Token usage tracking

Usage:
  from deepseek_api import DeepSeekClient
  client = DeepSeekClient()
  response = client.chat("Hello")
"""

import json
import os
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load .env from project root
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

# DeepSeek official API base
DEEPSEEK_BASE_URL = "https://api.deepseek.com"


class DeepSeekClient:
    """Client for DeepSeek API with retry and error handling."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or DEEPSEEK_API_KEY
        self.model = model or DEEPSEEK_MODEL
        self.base_url = DEEPSEEK_BASE_URL

        if not self.api_key or self.api_key == "your_deepseek_api_key_here":
            raise ValueError(
                "DeepSeek API key not set! "
                "Add DEEPSEEK_API_KEY to the .env file at the project root."
            )

        # Check if openai library is available
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key, base_url=f"{self.base_url}/v1")
            self._use_openai = True
        except ImportError:
            self._use_openai = False

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.3,
        max_tokens: int = 4096,
        retries: int = 3,
    ) -> str:
        """
        Send a chat request to DeepSeek with retry logic.

        Args:
            messages: List of {"role": ..., "content": ...} dicts.
            temperature: Sampling temperature (lower = more deterministic).
            max_tokens: Maximum tokens in the response.
            retries: Number of retries on failure.

        Returns:
            Response text content.
        """
        last_error = None

        for attempt in range(retries):
            try:
                if self._use_openai:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    return response.choices[0].message.content.strip()
                else:
                    # Fallback: direct HTTP call via requests
                    import requests as req

                    resp = req.post(
                        f"{self.base_url}/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": self.model,
                            "messages": messages,
                            "temperature": temperature,
                            "max_tokens": max_tokens,
                        },
                        timeout=120,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    return data["choices"][0]["message"]["content"].strip()

            except Exception as e:
                last_error = e
                wait = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                print(f"  [DeepSeek] Attempt {attempt + 1} failed: {e}")
                if attempt < retries - 1:
                    print(f"  [DeepSeek] Retrying in {wait}s...")
                    time.sleep(wait)

        raise RuntimeError(f"DeepSeek API failed after {retries} retries: {last_error}")

    def extract_json_array(self, text: str) -> list:
        """
        Extract a JSON array from DeepSeek's response text.
        DeepSeek usually outputs clean JSON, but this handles edge cases.
        """
        import re

        text = text.strip()

        # Try direct parse first
        try:
            result = json.loads(text)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

        # Find JSON array [...]
        match = re.search(r"\[.*?\]", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        # Find JSON object {...} and wrap in list
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return [json.loads(match.group())]
            except json.JSONDecodeError:
                pass

        return []
