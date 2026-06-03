import json
import os
import time
import requests

BASE_URL = "http://127.0.0.1:8000"
MODEL = "deepseek_v4"

TARGET_TOKENS = 16640  # > 128 * 128 = 16384，留一点余量
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "1"))


def count_tokens(prompt: str) -> int:
    resp = requests.post(
        f"{BASE_URL}/tokenize",
        json={"model": MODEL, "prompt": prompt},
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    return len(data["tokens"])


def tokenize(prompt: str) -> list[int]:
    resp = requests.post(
        f"{BASE_URL}/tokenize",
        json={"model": MODEL, "prompt": prompt},
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["tokens"]


def build_prompt() -> str:
    # 尽量使用英文重复片段，token 数更稳定；后面用 /tokenize 精确校准。
    unit = (
        "DeepSeek V4 prefix cache pooling validation. "
        "This sentence is repeated to build a long deterministic prefix. "
    )

    prompt = unit * 1000
    n = count_tokens(prompt)

    while n < TARGET_TOKENS:
        need_ratio = TARGET_TOKENS / max(n, 1)
        prompt += unit * max(100, int(100 * need_ratio))
        n = count_tokens(prompt)

    print(f"prompt tokens = {n}")
    return prompt


def request_once(prompt: str, idx: int):
    t0 = time.time()
    resp = requests.post(
        f"{BASE_URL}/v1/completions",
        json={
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": MAX_OUTPUT_TOKENS,
            "temperature": 0,
        },
        timeout=600,
    )
    latency = time.time() - t0
    print(f"request {idx}: status={resp.status_code}, latency={latency:.2f}s")
    print("full response:")
    try:
        data = resp.json()
        print(json.dumps(data, ensure_ascii=False, indent=2))
        for choice_idx, choice in enumerate(data.get("choices", [])):
            text = choice.get("text", "")
            output_tokens = tokenize(text) if text else []
            print(f"request {idx} choice {choice_idx} output text:")
            print(text)
            print(f"request {idx} choice {choice_idx} output token count = {len(output_tokens)}")
            print(f"request {idx} choice {choice_idx} output tokens:")
            print(output_tokens)
    except ValueError:
        print(resp.text)
    resp.raise_for_status()


if __name__ == "__main__":
    prompt = build_prompt()

    print("First request: should store KV into pool")
    request_once(prompt, 1)

    time.sleep(2)

    print("Second request: should hit prefix/KV pool")
    request_once(prompt, 2)
