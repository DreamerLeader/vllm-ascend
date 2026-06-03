import hashlib
import json
import os
import time
import requests

BASE_URL = "http://127.0.0.1:8000"
MODEL = "deepseek_v4"

TARGET_TOKENS = 16640  # > 128 * 128 = 16384，留一点余量
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "128"))
SEED = int(os.getenv("SEED", "0"))
DISABLE_THINKING = os.getenv("DISABLE_THINKING", "1") != "0"
QUESTION = os.getenv(
    "QUESTION",
    "Question: What is 123 + 456? Answer with only the final number.",
)


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

    prefix = unit * 1000
    n = count_tokens(prefix)

    while n < TARGET_TOKENS:
        need_ratio = TARGET_TOKENS / max(n, 1)
        prefix += unit * max(100, int(100 * need_ratio))
        n = count_tokens(prefix)

    prompt = f"{prefix}\n\n{QUESTION}\n"
    total_tokens = count_tokens(prompt)
    print(f"prefix tokens = {n}")
    print(f"prompt tokens = {total_tokens}")
    print("question:")
    print(QUESTION)
    return prompt


def build_payload(prompt: str) -> dict:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "min_tokens": 0,
        "stream": False,
        "n": 1,
        "best_of": 1,
        "seed": SEED,
        "temperature": 0,
        "top_p": 1.0,
        "top_k": 1,
        "min_p": 0.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "repetition_penalty": 1.0,
        "ignore_eos": False,
    }

    if DISABLE_THINKING:
        payload.update(
            {
                "enable_thinking": False,
                "thinking": {"type": "disabled"},
                "chat_template_kwargs": {"enable_thinking": False},
            }
        )

    return payload


def request_once(prompt: str, idx: int):
    payload = build_payload(prompt)
    payload_json = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    payload_sha256 = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()

    t0 = time.time()
    resp = requests.post(
        f"{BASE_URL}/v1/completions",
        json=payload,
        timeout=600,
    )
    latency = time.time() - t0
    print(f"request {idx}: status={resp.status_code}, latency={latency:.2f}s")
    print(f"request {idx} payload sha256 = {payload_sha256}")
    print(f"request {idx} deterministic seed = {SEED}")
    print(f"request {idx} disable thinking = {DISABLE_THINKING}")
    print("full response:")
    try:
        data = resp.json()
        print(json.dumps(data, ensure_ascii=False, indent=2))
        for choice_idx, choice in enumerate(data.get("choices", [])):
            text = choice.get("text", "")
            output_tokens = tokenize(text) if text else []
            print(f"request {idx} choice {choice_idx} answer:")
            print(text)
            print(f"request {idx} choice {choice_idx} output token count = {len(output_tokens)}")
            print(f"request {idx} choice {choice_idx} output tokens:")
            print(output_tokens)
    except ValueError:
        print(resp.text)
    resp.raise_for_status()
    return payload_sha256


if __name__ == "__main__":
    prompt = build_prompt()

    print("First request: should store KV into pool")
    first_payload_sha256 = request_once(prompt, 1)

    time.sleep(2)

    print("Second request: should hit prefix/KV pool")
    second_payload_sha256 = request_once(prompt, 2)

    print(f"same request payload = {first_payload_sha256 == second_payload_sha256}")
