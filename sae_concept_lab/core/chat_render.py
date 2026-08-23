"""Render a conversation through the MODEL'S OWN chat template, or refuse.

gemma-3-12b-it and Qwen3.5-27B are INSTRUCTION-TUNED. Handed a bare string
they do not answer it -- they continue it as a document. This product sent
the single word "hi" to gemma-3-12b-it and got back a JSON message-schema
tutorial, and sent it to Qwen and got a raw completion, because neither model
was ever asked anything: each was handed a fragment to complete. Every
generation this product had produced up to that point was a document
continuation wearing a chat UI.

The fix is not a system prompt and not a hand-written "User: ... Assistant:"
wrapper. It is the tokenizer's own `chat_template`, which ships with the model
and is part of its released identity -- the same conclusion qwen-sae-interp
reached for the science payload in d5d76fa.

TWO RULES, both learned the expensive way in that repository:

1. REFUSE, NEVER FALL BACK. A hand-rolled wrapper is a different prompt from
   the one the model was tuned on. Falling back to one would produce
   plausible-looking output from a render nobody ratified, which is worse than
   an error because it cannot be spotted by reading the output.

2. NEVER DOUBLE THE BOS. A chat template emits its own BOS, and both
   HookedTransformer.to_tokens and a HF tokenizer's add_special_tokens
   prepend another unless told not to. Two BOS tokens silently change the
   prompt the model sees (qwen-sae-interp 2f5bb39). Callers pass the
   template's output with prepending disabled, and check the result here.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


class ChatTemplateUnavailable(RuntimeError):
    """The loaded tokenizer publishes no chat template, so this product
    cannot render a conversation the way the model was tuned to receive one.
    Raised instead of falling back to an invented format."""


class DoubleBOSDetected(RuntimeError):
    """The token stream begins with two BOS tokens: the chat template emitted
    one and the tokenizer prepended another."""


def render_chat_prompt(
    tokenizer: Any,
    history: Sequence[tuple[str, str]],
    prompt: str,
    *,
    enable_thinking: bool | None = None,
) -> str:
    """The conversation as a single string, rendered by the model's own
    template with a generation prompt appended.

    `enable_thinking=False` suppresses Qwen3.5's reasoning trace, which would
    otherwise be emitted into the chat as if it were the answer. Templates
    that do not define the variable ignore it, which is why it is passed
    through rather than branched on by model name.
    """
    template = getattr(tokenizer, "chat_template", None)
    if not template:
        raise ChatTemplateUnavailable(
            "REFUSING TO GENERATE: the loaded tokenizer publishes no chat_template, so this "
            "conversation cannot be rendered the way the instruction-tuned model was trained to "
            "receive it. Refusing rather than substituting a hand-written wrapper: a wrapper is a "
            "different prompt, and its output would look plausible while being unattributable to "
            "any ratified render."
        )
    messages = [{"role": role, "content": content} for role, content in history]
    messages.append({"role": "user", "content": prompt})
    kwargs: dict[str, Any] = {}
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False, **kwargs
    )


def assert_at_most_one_leading_bos(token_ids: Sequence[int], bos_token_id: int | None) -> None:
    """Raise if the stream starts with the BOS token twice.

    A tokenizer that reports no BOS id cannot double one, so that case is a
    no-op rather than an assumption.
    """
    if bos_token_id is None:
        return
    ids = list(token_ids)
    if len(ids) >= 2 and ids[0] == bos_token_id and ids[1] == bos_token_id:
        raise DoubleBOSDetected(
            f"the rendered prompt begins with BOS token {bos_token_id} twice: the chat template "
            "emitted one and the tokenizer prepended another. Pass prepend_bos=False (Hooked"
            "Transformer.to_tokens) or add_special_tokens=False (HF tokenizer) when tokenizing "
            "already-templated text."
        )
