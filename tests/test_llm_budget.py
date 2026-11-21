import unittest
from unittest.mock import Mock

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


if __name__ == "__main__":
    unittest.main()
