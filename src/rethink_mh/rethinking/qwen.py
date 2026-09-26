"""Optional text-only Qwen2.5-Omni Thinker completion adapter."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


class QwenDependencyError(RuntimeError):
    """Raised when the optional Qwen runtime is unavailable."""


def load_pretrained_model_on_device(
    model_class: Any,
    model_path: str | Path,
    *,
    device: str,
    **from_pretrained_kwargs: Any,
) -> Any:
    """Load a pretrained model onto one requested runtime device."""

    model = model_class.from_pretrained(
        str(model_path),
        **from_pretrained_kwargs,
    )
    return model.to(device)


def first_complete_json_object(text: str) -> str | None:
    """Return a leading complete JSON object, excluding any later model chatter.

    Text before the object is deliberately rejected. This helper is used inside
    generation stopping, not by the workflow's general parser; other model adapters
    therefore remain subject to the exact-one-object contract.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    candidate = text.lstrip()
    if not candidate.startswith("{"):
        return None
    try:
        value, end = json.JSONDecoder().raw_decode(candidate)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict):
        return None
    return candidate[:end]


class QwenThinkerCompletionModel:
    """Generate one balanced JSON object from the Qwen Omni Thinker text path.

    Imports are lazy so textualization and workflow validation do not require PyTorch
    or Transformers. The Talker and all audio/video processor branches are bypassed;
    inputs are versioned evidence text chat messages only.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        max_new_tokens: int = 640,
        max_input_tokens: int = 7500,
    ) -> None:
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if max_input_tokens <= 0:
            raise ValueError("max_input_tokens must be positive")
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.max_input_tokens = max_input_tokens

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        *,
        adapter_path: str | Path | None = None,
        device: str = "cuda",
        dtype: str = "bfloat16",
        attention_implementation: str = "sdpa",
        max_new_tokens: int = 640,
        max_input_tokens: int = 7500,
    ) -> QwenThinkerCompletionModel:
        try:
            import torch
            from transformers import (
                AutoTokenizer,
                Qwen2_5OmniThinkerForConditionalGeneration,
            )
        except (ImportError, RuntimeError) as exc:  # pragma: no cover - optional runtime
            raise QwenDependencyError(
                "Qwen inference requires compatible torch and transformers installations"
            ) from exc

        if not hasattr(torch, dtype):
            raise ValueError(f"Unsupported torch dtype: {dtype}")
        torch_dtype = getattr(torch, dtype)
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), trust_remote_code=True
        )
        model = load_pretrained_model_on_device(
            Qwen2_5OmniThinkerForConditionalGeneration,
            model_path,
            device=device,
            torch_dtype=torch_dtype,
            attn_implementation=attention_implementation,
            low_cpu_mem_usage=True,
        )
        if adapter_path is not None:
            try:
                from peft import PeftModel
            except (ImportError, RuntimeError) as exc:  # pragma: no cover
                raise QwenDependencyError(
                    "Loading a Qwen adapter requires a compatible peft installation"
                ) from exc
            adapter = Path(adapter_path).expanduser().resolve()
            if not (adapter / "adapter_config.json").is_file():
                raise FileNotFoundError(f"Qwen adapter is incomplete: {adapter}")
            model = PeftModel.from_pretrained(model, str(adapter))
        model = model.eval()
        return cls(
            model,
            tokenizer,
            max_new_tokens=max_new_tokens,
            max_input_tokens=max_input_tokens,
        )

    def complete(
        self, messages: Sequence[Mapping[str, str]]
    ) -> str:
        try:
            import torch
            from transformers import StoppingCriteria, StoppingCriteriaList
        except (ImportError, RuntimeError) as exc:  # pragma: no cover - optional runtime
            raise QwenDependencyError(
                "Qwen inference requires compatible torch and transformers installations"
            ) from exc

        normalized = _validate_messages(messages)
        input_ids = self.tokenizer.apply_chat_template(
            normalized,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)
        input_length = int(input_ids.shape[-1])
        if input_length > self.max_input_tokens:
            raise ValueError(
                f"Qwen input has {input_length} tokens, exceeding the configured "
                f"limit of {self.max_input_tokens}"
            )

        tokenizer = self.tokenizer

        class _StopAfterJsonObject(StoppingCriteria):
            def __call__(
                self,
                generated_ids: Any,
                scores: Any,
                **kwargs: Any,
            ) -> bool:
                for row in generated_ids:
                    decoded = tokenizer.decode(
                        row[input_length:], skip_special_tokens=True
                    )
                    if first_complete_json_object(decoded) is None:
                        return False
                return True

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                use_cache=True,
                stopping_criteria=StoppingCriteriaList([_StopAfterJsonObject()]),
                pad_token_id=self.tokenizer.eos_token_id,
            )
        completion = self.tokenizer.decode(
            output_ids[0, input_length:], skip_special_tokens=True
        )
        complete_object = first_complete_json_object(completion)
        return complete_object if complete_object is not None else completion.strip()


class QwenAdapterDisabledCompletionModel:
    """Base-model completion view sharing one loaded PEFT-backed Qwen model.

    Evidence Literacy adapters are trained only for query and selection
    mechanics.  This view lets the initial and revision stages use the
    underlying base model without loading a second 7B checkpoint into GPU
    memory.  Calls are sequential; the PEFT adapter is disabled only for the
    duration of one completion.
    """

    def __init__(self, adapted_model: QwenThinkerCompletionModel) -> None:
        if not isinstance(adapted_model, QwenThinkerCompletionModel):
            raise TypeError(
                "adapted_model must be a QwenThinkerCompletionModel"
            )
        disable_adapter = getattr(
            adapted_model.model,
            "disable_adapter",
            None,
        )
        if not callable(disable_adapter):
            raise ValueError(
                "loaded Qwen model does not expose the PEFT disable_adapter "
                "context manager"
            )
        self.adapted_model = adapted_model
        self.tokenizer = adapted_model.tokenizer

    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
    ) -> str:
        disable_adapter = self.adapted_model.model.disable_adapter
        with disable_adapter():
            return self.adapted_model.complete(messages)


def _validate_messages(
    messages: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    if isinstance(messages, (str, bytes)) or not isinstance(messages, Sequence):
        raise TypeError("messages must be a sequence of role/content mappings")
    output: list[dict[str, str]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise TypeError(f"messages[{index}] must be a mapping")
        if set(message) != {"role", "content"}:
            raise ValueError(f"messages[{index}] must contain only role and content")
        role = message["role"]
        content = message["content"]
        if not isinstance(role, str) or not role:
            raise ValueError(f"messages[{index}].role must be non-empty text")
        if not isinstance(content, str) or not content:
            raise ValueError(f"messages[{index}].content must be non-empty text")
        output.append({"role": role, "content": content})
    if not output:
        raise ValueError("messages cannot be empty")
    return output


__all__ = [
    "QwenAdapterDisabledCompletionModel",
    "QwenDependencyError",
    "QwenThinkerCompletionModel",
    "first_complete_json_object",
    "load_pretrained_model_on_device",
]
