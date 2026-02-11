#!/usr/bin/env python3
"""
feedback_loop.py
-----------------

This module provides a simple automation harness for running a language model
in a feedback loop. The loop repeatedly calls an assistant model with an
initial prompt, then asks the model to critique its own response and
re‑answer the prompt based on the critique. This pattern can lead to
incrementally improved answers without human intervention.

Example usage (inside a git repo or standalone):

    from devkit.tools.feedback_loop import FeedbackLoop

    loop = FeedbackLoop(model="gpt-4-turbo", iterations=3)
    final_answer = loop.run("Explain the significance of the P=NP problem.")
    print(final_answer)

By default the OpenAI API key is read from the ``OPENAI_API_KEY`` environment
variable. See the README or docstring for additional configuration options.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

try:
    # Prefer the official openai package; fall back gracefully if missing.
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore


@dataclass
class FeedbackLoop:
    """Run an automated feedback loop around an LLM call.

    Parameters
    ----------
    model : str
        The model identifier to pass to the OpenAI API (e.g. ``"gpt-4-turbo"``).
    iterations : int
        Number of feedback iterations. Each iteration will produce a critique and
        an improved answer.
    temperature : float, optional
        Sampling temperature for the model. Lower values make answers more
        deterministic. Default is 0.3.
    max_tokens : int, optional
        Maximum tokens for each completion. Default is 1024.
    api_key : Optional[str], optional
        Override for the OpenAI API key. By default this is read from the
        ``OPENAI_API_KEY`` environment variable.
    """

    model: str = "gpt-4-turbo"
    iterations: int = 3
    temperature: float = 0.3
    max_tokens: int = 1024
    api_key: Optional[str] = None

    def __post_init__(self) -> None:
        if OpenAI is None:
            raise ImportError(
                "openai package is required for feedback loops. Please install it with pip."
            )
        key = self.api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "No OpenAI API key found. Set OPENAI_API_KEY environment variable or pass api_key."
            )
        # Construct a client instance. As of openai>=1.0.0 the API uses OpenAI() class.
        self.client = OpenAI(api_key=key)

    def _chat(self, messages: List[dict], **kwargs) -> str:
        """Internal helper to call the chat completion endpoint.

        Parameters
        ----------
        messages : List[dict]
            A list of chat messages in the OpenAI ChatCompletion format.

        Returns
        -------
        str
            The assistant's response content.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **kwargs,
        )
        choice = response.choices[0].message
        return choice.content.strip() if choice and choice.content else ""

    def run(self, prompt: str) -> str:
        """Execute the feedback loop and return the final answer.

        The loop operates as follows:

        1. Send the initial user prompt to the model and record the response.
        2. Ask the model to provide a self‑critique of its previous answer.
        3. Ask the model to produce an improved answer using the original prompt
           and the critique as guidance.
        4. Repeat steps 2–3 ``iterations`` times.

        Parameters
        ----------
        prompt : str
            The question or instruction to pass to the model.

        Returns
        -------
        str
            The answer produced after the final iteration.
        """
        # Start with the original prompt as the first user message.
        answer: str = ""
        for i in range(self.iterations):
            if i == 0:
                # Initial call: ask the model to answer the prompt directly.
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ]
                answer = self._chat(messages)
            else:
                # Reflective loop: ask for a critique of the last answer.
                critique_prompt = (
                    "You just answered the following question: \n"
                    f"\n{prompt}\n\n"
                    "Your answer was: \n"
                    f"\n{answer}\n\n"
                    "Please critique your own answer. Point out inaccuracies, omissions, or areas that could be improved."
                )
                critique_messages = [
                    {"role": "system", "content": "You are an expert reviewer."},
                    {"role": "user", "content": critique_prompt},
                ]
                critique = self._chat(critique_messages)

                # Now ask for an improved answer using the critique.
                improve_prompt = (
                    "Using the original question and the critique below, provide an improved answer.\n"
                    f"\nOriginal question:\n{prompt}\n"
                    f"\nPrevious answer:\n{answer}\n"
                    f"\nCritique:\n{critique}\n"
                )
                improve_messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": improve_prompt},
                ]
                answer = self._chat(improve_messages)
        return answer


def main() -> None:
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description=(
            "Run a feedback loop against OpenAI's Chat API. "
            "Provide an initial prompt and optionally override model and iterations."
        )
    )
    parser.add_argument(
        "prompt",
        help="The question or instruction to feed into the feedback loop.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4-turbo",
        help="OpenAI model name (default: gpt-4-turbo).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of feedback iterations to run (default: 3).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature (default: 0.3).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum number of tokens for each completion (default: 1024).",
    )
    parser.add_argument(
        "--api-key",
        help="OpenAI API key (defaults to OPENAI_API_KEY environment variable).",
    )
    args = parser.parse_args()

    loop = FeedbackLoop(
        model=args.model,
        iterations=args.iterations,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        api_key=args.api_key,
    )
    try:
        result = loop.run(args.prompt)
        print(result)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()