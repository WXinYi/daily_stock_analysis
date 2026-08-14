# -*- coding: utf-8 -*-
"""Tests for LLM thinking-mode extra_body resolution (src.agent.llm_adapter)."""

import unittest

from src.agent.llm_adapter import get_thinking_extra_body

_BOUNDED_PAYLOAD = {"thinking": {"type": "enabled", "budget_tokens": 4096}}


class ThinkingExtraBodyTest(unittest.TestCase):
    """get_thinking_extra_body resolves thinking payload per model."""

    def test_deepseek_v4_flash_gets_bounded_thinking(self):
        self.assertEqual(get_thinking_extra_body("deepseek-v4-flash"), _BOUNDED_PAYLOAD)

    def test_deepseek_v4_pro_gets_bounded_thinking(self):
        self.assertEqual(get_thinking_extra_body("deepseek-v4-pro"), _BOUNDED_PAYLOAD)

    def test_auto_thinking_model_returns_none(self):
        # deepseek-reasoner 自动思考，发 extra_body 会 400
        self.assertIsNone(get_thinking_extra_body("deepseek-reasoner"))

    def test_opt_in_model_returns_enabled_payload(self):
        self.assertEqual(
            get_thinking_extra_body("deepseek-chat"),
            {"thinking": {"type": "enabled"}},
        )

    def test_unrelated_model_returns_none(self):
        self.assertIsNone(get_thinking_extra_body("gpt-4o"))
        self.assertIsNone(get_thinking_extra_body(""))


if __name__ == "__main__":
    unittest.main()
