#!/usr/bin/env python3
"""Generate completions for FLASK questions using a vLLM-served model.

Reads a JSONL file of questions, sends each to the OpenAI-compatible API,
and writes the completions to an output JSONL file.

Supports async concurrency for throughput and auto-resume from partial output.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

import httpx


async def generate_one(
    client: httpx.AsyncClient,
    base_url: str,
    api_key: str,
    model: str,
    question: dict,
    max_tokens: int,
    temperature: float,
    use_completions_api: bool = False,
) -> dict:
    """Send one question to the API and return the result."""
    headers = {"Authorization": f"Bearer {api_key}"}

    if use_completions_api:
        # Base (non-chat) models: use /completions with raw prompt
        response = await client.post(
            f"{base_url}/completions",
            json={
                "model": model,
                "prompt": question["text"],
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            headers=headers,
            timeout=300.0,
        )
        response.raise_for_status()
        data = response.json()
        choice = data["choices"][0]
        completion_text = choice["text"]
    else:
        # Chat models: use /chat/completions with messages
        messages = [{"role": "user", "content": question["text"]}]
        response = await client.post(
            f"{base_url}/chat/completions",
            json={
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            headers=headers,
            timeout=300.0,
        )
        response.raise_for_status()
        data = response.json()
        choice = data["choices"][0]
        completion_text = choice["message"]["content"]

    return {
        "question_id": question["question_id"],
        "model": model,
        "text": question["text"],
        "reference_answer": question.get("answer", ""),
        "task": question.get("task", ""),
        "completion": completion_text,
        "finish_reason": choice.get("finish_reason", ""),
        "usage": data.get("usage", {}),
    }


async def main():
    parser = argparse.ArgumentParser(description="Generate completions for FLASK questions")
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument("--model", required=True, help="Model name (as served by vLLM)")
    parser.add_argument("--base-url", required=True, help="vLLM API base URL")
    parser.add_argument("--api-key", default="not-needed", help="API key")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--completions-api", action="store_true",
                        help="Use /completions instead of /chat/completions (for base models)")
    args = parser.parse_args()

    # Load questions
    questions = []
    with open(args.input) as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    print(f"Loaded {len(questions)} questions from {args.input}")

    # Check for existing completions (resume support)
    done_ids = set()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    done_ids.add(row["question_id"])
        print(f"Resuming: {len(done_ids)} already completed")

    remaining = [q for q in questions if q["question_id"] not in done_ids]
    print(f"Generating {len(remaining)} completions (concurrency={args.concurrency})")

    if not remaining:
        print("All questions already completed!")
        return

    # Generate with async concurrency
    semaphore = asyncio.Semaphore(args.concurrency)
    completed = 0
    errors = 0
    start_time = time.monotonic()

    async with httpx.AsyncClient() as client:
        outfile = open(output_path, "a")

        async def process(q):
            nonlocal completed, errors
            async with semaphore:
                try:
                    result = await generate_one(
                        client, args.base_url, args.api_key, args.model,
                        q, args.max_tokens, args.temperature,
                        use_completions_api=args.completions_api,
                    )
                    outfile.write(json.dumps(result) + "\n")
                    outfile.flush()
                    completed += 1
                    if completed % 10 == 0:
                        elapsed = time.monotonic() - start_time
                        rate = completed / elapsed
                        print(f"  {completed}/{len(remaining)} ({rate:.1f}/s)", flush=True)
                except Exception as e:
                    errors += 1
                    print(f"  ERROR on q{q['question_id']}: {e}", flush=True)

        tasks = [asyncio.create_task(process(q)) for q in remaining]
        await asyncio.gather(*tasks)

        outfile.close()

    elapsed = time.monotonic() - start_time
    print(f"\nDone: {completed} completed, {errors} errors in {elapsed:.0f}s")


if __name__ == "__main__":
    asyncio.run(main())
