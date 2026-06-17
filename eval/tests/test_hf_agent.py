import unittest

from model.hf_agent import HuggingFaceAgent


class FakeTokenizer:
    chat_template = None


class HuggingFaceAgentMessageTests(unittest.TestCase):
    def make_agent(self, image_policy="error"):
        agent = object.__new__(HuggingFaceAgent)
        agent.image_policy = image_policy
        agent.tokenizer = FakeTokenizer()
        return agent

    def test_normalizes_openai_style_text_parts(self):
        agent = self.make_agent()

        messages = [
            {"role": "system", "content": "You are a helpful medical assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Q: Which option is correct?"},
                ],
            },
        ]

        normalized = agent._normalize_messages(messages)

        self.assertEqual(
            normalized,
            [
                {"role": "system", "content": "You are a helpful medical assistant."},
                {"role": "user", "content": "Q: Which option is correct?"},
            ],
        )

    def test_text_only_agent_rejects_image_inputs_by_default(self):
        agent = self.make_agent()

        with self.assertRaisesRegex(ValueError, "text-only"):
            agent._content_to_text(
                [
                    {"type": "text", "text": "Question text"},
                    {"type": "image_url", "image_url": {"url": "images/MM-1.jpeg"}},
                ]
            )

    def test_describe_image_policy_omits_image_without_base64_payload(self):
        agent = self.make_agent(image_policy="describe")

        rendered = agent._content_to_text(
            [
                {"type": "text", "text": "Question text"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,abcdef"}},
            ]
        )

        self.assertIn("Question text", rendered)
        self.assertIn("embedded image", rendered)
        self.assertNotIn("abcdef", rendered)

    def test_plain_prompt_fallback_uses_chat_roles(self):
        agent = self.make_agent()

        prompt = agent._render_prompt(
            [
                {"role": "system", "content": "System message"},
                {"role": "user", "content": "User message"},
            ]
        )

        self.assertEqual(prompt, "System: System message\nUser: User message\nAssistant:")


if __name__ == "__main__":
    unittest.main()
