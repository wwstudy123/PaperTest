import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from admtune.llm_check import check_endpoint, main


class LLMCheckTests(unittest.TestCase):
    def test_single_small_request_and_explicit_env_precedence(self):
        with tempfile.TemporaryDirectory() as folder:
            env = Path(folder) / ".env"
            env.write_text(
                "OPENAI_API_KEY=test-file-key\nOPENAI_BASE_URL=https://example.invalid/v1\n"
                "OPENAI_MODEL=fixture\nOPENAI_ENABLE_THINKING=false\n",
            )
            response = Mock()
            response.status_code = 200
            response.json.return_value = {
                "usage": {"total_tokens": 43},
                "choices": [{"message": {"content": '{"status":"ok"}'}}],
            }
            with patch("requests.post", return_value=response) as post:
                result = check_endpoint(env, Path(folder) / "output")
            self.assertTrue(result["valid"])
            self.assertEqual(post.call_count, 1)
            self.assertEqual(post.call_args.kwargs["json"]["max_tokens"], 256)
            self.assertFalse(post.call_args.kwargs["json"]["enable_thinking"])
            text = (Path(folder) / "output/summary.json").read_text()
            self.assertNotIn("test-file-key", text)

    def test_error_does_not_retry_and_redacts_key(self):
        with tempfile.TemporaryDirectory() as folder:
            env = Path(folder) / ".env"
            env.write_text("OPENAI_API_KEY=secret-fixture\nOPENAI_BASE_URL=https://example.invalid\nOPENAI_MODEL=fixture\n")
            with patch("requests.post", side_effect=RuntimeError("secret-fixture")) as post:
                result = check_endpoint(env, Path(folder) / "output")
            self.assertFalse(result["valid"])
            self.assertEqual(post.call_count, 1)
            self.assertNotIn("secret-fixture", json.dumps(result))

    def test_missing_key_stops_before_request(self):
        with tempfile.TemporaryDirectory() as folder:
            env = Path(folder) / ".env"
            env.write_text("OPENAI_API_KEY=\nOPENAI_BASE_URL=\nOPENAI_MODEL=\n")
            with patch("requests.post") as post:
                with self.assertRaises(ValueError):
                    check_endpoint(env, Path(folder) / "output")
            post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
