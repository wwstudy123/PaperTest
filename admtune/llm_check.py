"""一次、低输出上限的模型接口检查；不访问数据库、不隐式重试。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

from dotenv import dotenv_values
import requests


def check_endpoint(env_file: Path, output: Path) -> dict:
    values = {**os.environ, **{k: v for k, v in dotenv_values(env_file).items() if v is not None}}
    key, base, model = [values.get(k, "") for k in ["OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL"]]
    if not all([key, base, model]):
        raise ValueError("请先填写密钥、接口基础地址和模型名")
    output.mkdir(parents=True, exist_ok=False)
    body = {
        "model": model, "max_tokens": 256, "temperature": 0.2,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": "只返回一个 JSON 对象。"},
            {"role": "user", "content": '请仅返回 {"status":"ok"}，不要解释。'},
        ],
    }
    if values.get("OPENAI_ENABLE_THINKING", "").strip():
        body["enable_thinking"] = values["OPENAI_ENABLE_THINKING"].lower() in {"1", "true", "yes"}
    started = time.monotonic()
    result = {"model": model, "base_url": base, "api_calls": 1, "output_token_limit": 256}
    try:
        response = requests.post(
            base.rstrip("/") + "/chat/completions", json=body,
            headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
            timeout=120,
        )
        result["http_status"] = response.status_code
        response.raise_for_status()
        data = response.json()
        result["usage"] = data.get("usage")
        content = data["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        result["response"] = parsed
        result["valid"] = parsed == {"status": "ok"}
    except Exception as exc:
        result.update(valid=False, error=str(exc).replace(key, "<redacted>"))
    result["elapsed_seconds"] = time.monotonic() - started
    serialized = json.dumps(result, ensure_ascii=False, indent=2).replace(key, "<redacted>")
    (output / "summary.json").write_text(serialized + "\n", encoding="utf-8")
    return json.loads(serialized)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-paid-llm", action="store_true")
    args = parser.parse_args()
    if not args.allow_paid_llm:
        parser.error("接口检查会产生一次模型调用，必须显式传入 --allow-paid-llm")
    result = check_endpoint(args.env_file, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
