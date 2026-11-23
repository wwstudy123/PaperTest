import unittest
from unittest.mock import Mock
from pathlib import Path
import tempfile
import json

from admtune.agents import AgenticDBBackend
from admtune.budget import BudgetExhausted, Ledger, Limits


class LLMBudgetTests(unittest.TestCase):
    def client(self):
        client = Mock()
        client.dry_run = False
        client.api_key = "not-a-real-key"
        client.model = "fixture"
        client.timeout_s = 120
        client._call_openai_compatible.return_value = {"selected": ["buffer"]}
        client.consume_last_usage.return_value = {"total_tokens": 50}
        return client

    def test_exact_requests_cached_and_retries_disabled(self):
        client = self.client()
        ledger = Ledger(Limits(llm_calls=1, llm_tokens=10000))
        backend = AgenticDBBackend(client, ledger, max_output_tokens=100)
        a = backend.complete("selector", {"catalog": []})
        b = backend.complete("selector", {"catalog": []})
        self.assertEqual(a, b)
        self.assertEqual(client._call_openai_compatible.call_count, 1)
        self.assertEqual(ledger.llm_calls, 1)
        self.assertEqual(ledger.llm_tokens, 50)
        self.assertEqual(client.response_retries, 0)

    def test_compact_transport_preserves_payload_and_byte_reservation(self):
        from admtune.agents import PROMPTS
        client = self.client()
        client.consume_last_usage.return_value = None
        ledger = Ledger(Limits(llm_calls=1, llm_tokens=10000))
        backend = AgenticDBBackend(client, ledger, max_output_tokens=100)
        payload = {"catalog": [{"name": "缓冲池", "values": [1, 2]}]}
        backend.complete("selector", payload)
        text = client._call_openai_compatible.call_args.args[0]
        self.assertEqual(json.loads(text), payload)
        self.assertEqual(text, json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                         separators=(",", ":")))
        self.assertEqual(ledger.llm_tokens, len((PROMPTS["selector"] + text).encode()) + 164)

    def test_error_consumes_attempt_and_does_not_fallback(self):
        client = self.client()
        client._call_openai_compatible.side_effect = RuntimeError("API failed")
        client.consume_last_usage.return_value = None
        ledger = Ledger(Limits(llm_calls=1, llm_tokens=10000))
        backend = AgenticDBBackend(client, ledger, max_output_tokens=100)
        with self.assertRaises(RuntimeError):
            backend.complete("selector", {})
        self.assertGreater(ledger.llm_tokens, 0)
        with self.assertRaises(BudgetExhausted):
            backend.complete("selector", {})

    def test_audit_trace_has_usage_but_never_credentials(self):
        client = self.client()
        with tempfile.TemporaryDirectory() as folder:
            backend = AgenticDBBackend(
                client, Ledger(Limits(llm_calls=1, llm_tokens=10000)),
                trace_dir=Path(folder),
            )
            backend.complete("selector", {"catalog": []})
            text = (Path(folder) / "001-selector.json").read_text()
            self.assertNotIn(client.api_key, text)
            self.assertEqual(json.loads(text)["usage"]["total_tokens"], 50)
            self.assertEqual(json.loads(text)["response"], {"selected": ["buffer"]})


if __name__ == "__main__":
    unittest.main()
