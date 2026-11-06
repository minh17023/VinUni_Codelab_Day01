"""Completed solutions for the Day 1 LLM API lab.

The provider SDKs are imported inside the functions so unit tests can replace
their clients with mocks without making network requests.
"""

import os
import time
from typing import Any, Callable


PRICING_1M_TOKENS = {
    "gpt-4o": {"input": 5.00, "output": 20.00},
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "gemini-2.5-flash": {"input": 0.075, "output": 0.300},
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku": {"input": 0.80, "output": 4.00},
}

OPENAI_MODEL = "gpt-4o"
OPENAI_MINI_MODEL = "gpt-4o-mini"
GEMINI_MODEL = "gemini-2.5-flash"
ANTHROPIC_MODEL = "claude-3-5-haiku"


def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """Return OpenAI response text, request latency and token counts."""
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    latency = time.perf_counter() - start
    usage = {
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
    }
    return response.choices[0].message.content or "", latency, usage


def call_gemini(
    prompt: str,
    model: str = GEMINI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """Return Gemini response text, request latency and token counts."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    config = types.GenerateContentConfig(
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_tokens,
    )
    start = time.perf_counter()
    response = client.models.generate_content(
        model=model, contents=prompt, config=config
    )
    latency = time.perf_counter() - start
    usage = {
        "input_tokens": response.usage_metadata.prompt_token_count,
        "output_tokens": response.usage_metadata.candidates_token_count,
    }
    return response.text or "", latency, usage


def call_anthropic(
    prompt: str,
    model: str = ANTHROPIC_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """Return Claude response text, request latency and token counts."""
    import anthropic

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    start = time.perf_counter()
    response = client.messages.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    latency = time.perf_counter() - start
    text = "".join(part.text for part in response.content if isinstance(part.text, str))
    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }
    return text, latency, usage


def compare_models(prompt: str) -> dict:
    """Compare the three lab models using the lab's fixed price estimates."""
    calls = (
        ("gpt4o", OPENAI_MODEL, call_openai(prompt, model=OPENAI_MODEL)),
        (
            "gpt4o_mini",
            OPENAI_MINI_MODEL,
            call_openai(prompt, model=OPENAI_MINI_MODEL),
        ),
        ("gemini_flash", GEMINI_MODEL, call_gemini(prompt, model=GEMINI_MODEL)),
    )
    comparison = {}
    for name, model, (response, latency, usage) in calls:
        rates = PRICING_1M_TOKENS[model]
        input_tokens = usage["input_tokens"]
        output_tokens = usage["output_tokens"]
        comparison[name] = {
            "response": response,
            "latency": latency,
            "cost": (
                input_tokens * rates["input"] + output_tokens * rates["output"]
            ) / 1_000_000,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
    return comparison


def streaming_chatbot() -> None:
    """Stream a Gemini chat in the terminal, retaining three completed turns."""
    from google import genai
    from google.genai import types

    client = None
    history = []
    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if prompt.lower() in {"quit", "exit"}:
            return
        if not prompt:
            continue

        if client is None:
            client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        contents = history + [types.Content(role="user", parts=[types.Part(text=prompt)])]
        chunks = client.models.generate_content_stream(
            model=GEMINI_MODEL, contents=contents
        )
        answer_parts = []
        print("Gemini: ", end="", flush=True)
        for chunk in chunks:
            piece = chunk.text or ""
            if piece:
                print(piece, end="", flush=True)
                answer_parts.append(piece)
        print()
        history.extend(
            [
                types.Content(role="user", parts=[types.Part(text=prompt)]),
                types.Content(
                    role="model", parts=[types.Part(text="".join(answer_parts))]
                ),
            ]
        )
        history = history[-6:]


def retry_with_backoff(
    fn: Callable[[], Any], max_retries: int = 3, base_delay: float = 0.1
) -> Any:
    """Call fn once, then retry failures with exponentially increasing waits."""
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception:
            if attempt == max_retries:
                raise
            time.sleep(base_delay * (2 ** attempt))


def batch_compare(prompts: list[str]) -> list[dict]:
    """Run a comparison for each prompt while preserving input order."""
    return [{**compare_models(prompt), "prompt": prompt} for prompt in prompts]


def format_comparison_table(results: list[dict]) -> str:
    """Format batch comparison results as a Markdown table."""
    def clean(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")

    rows = [
        "| Prompt | Model | Response (truncated) | Latency | Tokens (In/Out) | Cost (USD) |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    names = (
        ("gpt4o", "GPT-4o"),
        ("gpt4o_mini", "GPT-4o-Mini"),
        ("gemini_flash", "Gemini-Flash"),
    )
    for result in results:
        for key, label in names:
            stats = result[key]
            response = str(stats["response"])
            shortened = response[:50] + ("..." if len(response) > 50 else "")
            rows.append(
                f"| {clean(result['prompt'])} | {label} | {clean(shortened)} | "
                f"{stats['latency']:.2f}s | "
                f"{stats['input_tokens']}/{stats['output_tokens']} | "
                f"${stats['cost']:.8f} |"
            )
    return "\n".join(rows)
