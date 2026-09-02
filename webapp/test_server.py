"""Fast regression tests; no checkpoint loading or GPU required."""
import io
import unittest
from types import SimpleNamespace

from webapp.server import (
    Handler, MAX_REQUEST_BYTES, _params_for_model, clean_params, parse_generated,
)


class ServerTests(unittest.TestCase):
    def read_body(self, raw, length=None):
        request = SimpleNamespace(
            headers={"Content-Length": str(len(raw) if length is None else length)},
            rfile=io.BytesIO(raw), close_connection=False,
        )
        return Handler._read_json_body(request)

    def test_valid_json(self):
        self.assertEqual(self.read_body(b'{"problem":"2+2"}'), {"problem": "2+2"})

    def test_invalid_json_or_non_object(self):
        for raw in (b"[]", b"null", b"bad", b"\xff"):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                self.read_body(raw)

    def test_invalid_lengths(self):
        for length in (-1, "invalid"):
            with self.subTest(length=length), self.assertRaises(ValueError):
                self.read_body(b"", length)

    def test_oversized_request(self):
        with self.assertRaises(OverflowError):
            self.read_body(b"", MAX_REQUEST_BYTES + 1)

    def test_decoding_bounds(self):
        params = clean_params({"temperature": 99, "max_new_tokens": 9999, "top_k": -1})
        self.assertEqual(params["temperature"], 5.0)
        self.assertEqual(params["max_new_tokens"], 1024)
        self.assertIsNone(params["top_k"])

    def test_top_k_cannot_exceed_vocabulary(self):
        model = SimpleNamespace(tokenizer=SimpleNamespace(vocab_size=56))
        source = {"top_k": 100}
        self.assertEqual(_params_for_model(source, model)["top_k"], 56)
        self.assertEqual(source["top_k"], 100)

    def test_final_answer_is_last_step(self):
        parsed = parse_generated("Problem: 2+2; step: 2 + 2; step: 4")
        self.assertEqual(parsed["final"], "4")
        self.assertEqual(parsed["steps"], ["2 + 2", "4"])


if __name__ == "__main__":
    unittest.main()
