import logging
import os
from typing import Any, List

from model.base_agent import LLMAgent


class HuggingFaceAgent(LLMAgent):
    """Local/remote Hugging Face causal-LM agent for text-only MedXpertQA runs."""

    IMAGE_POLICIES = {"error", "ignore", "describe"}

    def __init__(
        self,
        model_name: str,
        temperature: float = 0,
        device_map: str | None = "auto",
        torch_dtype: str | None = "auto",
        max_new_tokens: int | None = None,
        max_input_tokens: int | None = None,
        trust_remote_code: bool = False,
        local_files_only: bool = False,
        revision: str | None = None,
        cache_dir: str | None = None,
        attn_implementation: str | None = None,
        image_policy: str = "error",
        token_env: str = "HF_TOKEN",
        seed: int = 0,
    ) -> None:
        super().__init__(model_name, temperature)
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.max_new_tokens = max_new_tokens or self._default_max_new_tokens(model_name)
        self.max_input_tokens = max_input_tokens
        self.trust_remote_code = trust_remote_code
        self.local_files_only = local_files_only
        self.revision = revision
        self.cache_dir = cache_dir
        self.attn_implementation = attn_implementation
        self.image_policy = image_policy
        self.token_env = token_env
        self.seed = seed

        if self.image_policy not in self.IMAGE_POLICIES:
            raise ValueError(
                f"Unsupported Hugging Face image policy: {self.image_policy}. "
                f"Choose one of {sorted(self.IMAGE_POLICIES)}."
            )

        self._load_model()

    @staticmethod
    def _default_max_new_tokens(model_name: str) -> int:
        lowered = model_name.lower()
        if any(marker in lowered for marker in ("reasoning", "reasoner", "r1", "qwq", "qvq")):
            return 8192
        return 2048

    @staticmethod
    def _resolve_dtype(torch_module: Any, torch_dtype: str | None) -> Any:
        if torch_dtype in (None, "", "none", "default"):
            return None
        if torch_dtype == "auto":
            return "auto"

        dtype_map = {
            "bfloat16": torch_module.bfloat16,
            "bf16": torch_module.bfloat16,
            "float16": torch_module.float16,
            "fp16": torch_module.float16,
            "float32": torch_module.float32,
            "fp32": torch_module.float32,
        }
        try:
            return dtype_map[torch_dtype.lower()]
        except KeyError as exc:
            raise ValueError(
                "Unsupported --hf-torch-dtype value. Use auto, bfloat16, float16, "
                "float32, none, or default."
            ) from exc

    def _load_model(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
        except ImportError as exc:
            raise RuntimeError(
                "Hugging Face inference requires extra dependencies. "
                "Install them from eval/requirements-hf.txt."
            ) from exc

        self.torch = torch
        set_seed(self.seed)

        token = self._get_hf_token()
        shared_kwargs: dict[str, Any] = {
            "trust_remote_code": self.trust_remote_code,
            "local_files_only": self.local_files_only,
        }
        if token:
            shared_kwargs["token"] = token
        if self.revision:
            shared_kwargs["revision"] = self.revision
        if self.cache_dir:
            shared_kwargs["cache_dir"] = self.cache_dir

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, **shared_kwargs)

        model_kwargs = dict(shared_kwargs)
        resolved_dtype = self._resolve_dtype(torch, self.torch_dtype)
        if resolved_dtype is not None:
            model_kwargs["torch_dtype"] = resolved_dtype
        if self.device_map not in (None, "", "none"):
            if self.device_map == "cpu":
                model_kwargs["device_map"] = {"": "cpu"}
            else:
                model_kwargs["device_map"] = self.device_map
        if self.attn_implementation:
            model_kwargs["attn_implementation"] = self.attn_implementation
        model_kwargs["low_cpu_mem_usage"] = True

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        self.model.eval()

        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _get_hf_token(self) -> str | None:
        env_names = [self.token_env, "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"]
        for env_name in env_names:
            if env_name:
                token = os.environ.get(env_name)
                if token:
                    return token
        return None

    def get_response(self, messages: List[dict]) -> tuple[str, list]:
        text_messages = self._normalize_messages(messages)
        prompt = self._render_prompt(text_messages)
        tokenizer_kwargs: dict[str, Any] = {"return_tensors": "pt"}
        if self.max_input_tokens:
            tokenizer_kwargs.update({"truncation": True, "max_length": self.max_input_tokens})

        inputs = self.tokenizer(prompt, **tokenizer_kwargs)
        input_length = inputs["input_ids"].shape[-1]
        input_device = self._input_device()
        inputs = {key: value.to(input_device) for key, value in inputs.items()}

        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if self.temperature > 0:
            generation_kwargs["temperature"] = self.temperature

        with self.torch.inference_mode():
            output_ids = self.model.generate(**inputs, **generation_kwargs)

        generated_ids = output_ids[0][input_length:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return response, []

    def _input_device(self) -> Any:
        if hasattr(self.model, "hf_device_map"):
            for device in self.model.hf_device_map.values():
                if device != "disk":
                    return self.torch.device(device)
        if hasattr(self.model, "device"):
            return self.model.device
        return self.torch.device("cpu")

    def _render_prompt(self, messages: list[dict[str, str]]) -> str:
        if getattr(self.tokenizer, "chat_template", None):
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception as exc:
                logging.warning(
                    "Falling back to plain prompt rendering because chat template failed: %s",
                    exc,
                )

        rendered = []
        for message in messages:
            role = message["role"].capitalize()
            rendered.append(f"{role}: {message['content']}")
        rendered.append("Assistant:")
        return "\n".join(rendered)

    def _normalize_messages(self, messages: List[dict]) -> list[dict[str, str]]:
        normalized = []
        for message in messages:
            content = self._content_to_text(message.get("content", ""))
            if content.strip():
                normalized.append(
                    {
                        "role": message.get("role", "user"),
                        "content": content,
                    }
                )
        return normalized

    def _content_to_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return str(content)

        chunks = []
        for part in content:
            part_type = part.get("type") if isinstance(part, dict) else None
            if part_type == "text":
                chunks.append(part.get("text", ""))
            elif part_type == "image_url":
                image_text = self._handle_image_part(part)
                if image_text:
                    chunks.append(image_text)
            elif part is not None:
                chunks.append(str(part))
        return "\n".join(chunk for chunk in chunks if chunk)

    def _handle_image_part(self, part: dict) -> str:
        if self.image_policy == "ignore":
            return ""
        if self.image_policy == "error":
            raise ValueError(
                "This Hugging Face agent is text-only and received image input. "
                "Use --task text, switch to a multimodal model integration, or set "
                "--hf-image-policy describe/ignore only for non-benchmark smoke tests."
            )

        image_url = part.get("image_url", {})
        if isinstance(image_url, dict):
            image_ref = image_url.get("url", "image input")
        else:
            image_ref = str(image_url)
        if image_ref.startswith("data:image"):
            image_ref = "embedded image"
        return f"[Image omitted from text-only Hugging Face prompt: {image_ref}]"

    def image_content(self, img_path: str) -> dict:
        return {"type": "image_url", "image_url": {"url": img_path.strip()}}
