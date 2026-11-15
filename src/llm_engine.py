
"""Interface for generating text with the supplied LLM SDK."""

import json

from llm_sdk import Small_LLM_Model


class LLMEngine:
    """Manage model initialization and text generation."""

    def __init__(self) -> None:
        """Initialize the language model once."""
        self.model = Small_LLM_Model()

    def generate(
        self,
        prompt: str,
        max_tokens: int = 100
    ) -> str:
        """Generate text using greedy next-token selection."""
        input_ids = self.model.encode(prompt).tolist()[0]
        generated_ids: list[int] = []

        for _ in range(max_tokens):
            logits = self.model.get_logits_from_input_ids(
                input_ids
            )

            next_id = logits.index(max(logits))

            input_ids.append(next_id)
            generated_ids.append(next_id)

            generated_text = self.model.decode(generated_ids)

            try:
                result = json.loads(generated_text)
            except json.JSONDecodeError:
                continue

            if isinstance(result, dict):
                return generated_text

        raise ValueError(
            "Model did not generate complete JSON"
        )
