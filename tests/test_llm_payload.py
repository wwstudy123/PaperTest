import unittest
from unittest.mock import Mock, patch

import requests

from admtune.agents import PROMPTS
from llm_client import LLMClient


class LLMPayloadTests(unittest.TestCase):
    def test_every_structured_role_explicitly_requests_json(self):
        for role, prompt in PROMPTS.items():
            self.assertIn("json", prompt.lower(), role)

    def test_compatible_client_adds_json_keyword_when_omitted(self):
        client = LLMClient("fixture-secret", "https://example.invalid/v1", "fixture", response_retries=0)
        response = Mock()
        response.json.return_value = {"choices": [{"message": {"content": '{"ok":true}'}}], "usage": {"total_tokens": 5}}
        with patch("requests.post", return_value=response) as post:
            result = client._call_openai_compatible("test", operation="test", system_prompt="返回结构化结果")
        self.assertTrue(result["ok"])
        self.assertIn("json", post.call_args.kwargs["json"]["messages"][0]["content"].lower())

    def test_api_failure_preserves_reason_but_redacts_credentials(self):
        client = LLMClient("fixture-secret", "https://example.invalid/v1", "fixture", response_retries=0)
        response = Mock()
        response.status_code = 400
        response.text = "messages must contain json; fixture-secret"
        response.raise_for_status.side_effect = requests.HTTPError("bad request", response=response)
        with patch("requests.post", return_value=response) as post:
            with self.assertRaises(requests.HTTPError) as exc:
                client._call_openai_compatible("test", operation="test", system_prompt="JSON")
        self.assertIn("messages must contain json", str(exc.exception))
        self.assertNotIn("fixture-secret", str(exc.exception))
        self.assertEqual(post.call_count, 1)


if __name__ == "__main__":
    unittest.main()
