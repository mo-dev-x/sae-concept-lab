"""Conversations must reach an instruction-tuned model through the model's
own chat template.

WHY THIS FILE EXISTS. Both backends tokenized `request.prompt` directly. Sent
the single word "hi", gemma-3-12b-it returned a JSON message-schema tutorial
and Qwen returned a raw completion -- neither had been asked anything, each
had been handed a document fragment to continue. Every generation this
product ever produced was a continuation wearing a chat UI, and 345 tests
were green throughout, because the fake tokenizer had no chat template to
skip. A fake more permissive than the real thing cannot fail this way.
"""

from __future__ import annotations

import pytest

from sae_concept_lab.core.chat_render import (
    ChatTemplateUnavailable,
    DoubleBOSDetected,
    assert_at_most_one_leading_bos,
    render_chat_prompt,
)
from sae_concept_lab.core.protocol import GenerationRequest
from tests._fake_runtime import FakeGemmaModel, FakeTokenizer, install_fake_torch


class _NoTemplateTokenizer:
    chat_template = None

    def apply_chat_template(self, *a, **k):  # pragma: no cover - must never run
        raise AssertionError("must refuse before reaching the tokenizer")


# ---------------------------------------------------------------------------
# The renderer
# ---------------------------------------------------------------------------


def test_render_goes_through_the_tokenizers_own_template_with_history_and_a_generation_prompt():
    tok = FakeTokenizer()
    rendered = render_chat_prompt(tok, (("user", "first"), ("assistant", "reply")), "second")

    roles = [m["role"] for m in tok.last_messages]
    contents = [m["content"] for m in tok.last_messages]
    assert roles == ["user", "assistant", "user"]
    assert contents == ["first", "reply", "second"]
    # add_generation_prompt=True: the model is handed the start of ITS turn,
    # which is the difference between answering and continuing.
    assert rendered.endswith("<assistant>")


def test_a_tokenizer_with_no_chat_template_refuses_instead_of_falling_back():
    """The dangerous failure is not an exception, it is a hand-rolled
    'User: ... Assistant:' wrapper: a different prompt from the one the model
    was tuned on, whose output looks plausible and is attributable to no
    ratified render. So the absence of a template must raise."""
    with pytest.raises(ChatTemplateUnavailable) as excinfo:
        render_chat_prompt(_NoTemplateTokenizer(), (), "hi")
    assert "REFUSING TO GENERATE" in str(excinfo.value)


def test_an_empty_string_template_is_absent_too():
    tok = FakeTokenizer()
    tok.chat_template = ""
    with pytest.raises(ChatTemplateUnavailable):
        render_chat_prompt(tok, (), "hi")


def test_enable_thinking_is_passed_through_to_the_template():
    """Qwen3.5 emits a reasoning trace into the reply unless the template is
    told not to. Templates without the variable ignore it, which is why this
    is passed through rather than branched on by model name."""
    tok = FakeTokenizer()
    render_chat_prompt(tok, (), "hi", enable_thinking=False)
    assert tok.last_apply_kwargs == {"enable_thinking": False}

    tok2 = FakeTokenizer()
    render_chat_prompt(tok2, (), "hi")
    assert tok2.last_apply_kwargs == {}


# ---------------------------------------------------------------------------
# The BOS guard, tested directly: the fake vocabulary has no real BOS, so
# pretending at one through a fake would prove nothing.
# ---------------------------------------------------------------------------


def test_two_leading_bos_tokens_are_refused():
    with pytest.raises(DoubleBOSDetected):
        assert_at_most_one_leading_bos([2, 2, 105, 106], bos_token_id=2)


def test_one_leading_bos_is_fine_and_so_is_a_repeat_further_in():
    # The guard returns nothing; the behaviour under test is that it does NOT
    # raise. Comparing the return to None would assert precisely nothing --
    # ruff B015 caught exactly that mistake in the first draft of this file.
    assert_at_most_one_leading_bos([2, 105, 2, 106], bos_token_id=2)
    assert_at_most_one_leading_bos([105, 106], bos_token_id=2)
    assert_at_most_one_leading_bos([], bos_token_id=2)


def test_a_tokenizer_reporting_no_bos_id_cannot_double_one():
    assert_at_most_one_leading_bos([7, 7, 7], bos_token_id=None)


# ---------------------------------------------------------------------------
# The wiring: it is not enough that a renderer exists.
# ---------------------------------------------------------------------------


def _baseline_request(prompt: str, history=()) -> GenerationRequest:
    return GenerationRequest(
        history=tuple(history),
        prompt=prompt,
        model_key="gemma",
        decoding={"max_new_tokens": 4},
        seed=0,
        apply_intervention=False,
        resolved_config=None,
    )


def _install_gemma_loader(monkeypatch) -> FakeGemmaModel:
    install_fake_torch(monkeypatch)
    model = FakeGemmaModel()
    provenance = {
        "target": "gemma-3-12b-it",
        "model": {"repository": "google/gemma-3-12b-it", "actual_class": "HookedTransformer"},
        "sae": {"repository": "google/gemma-scope-2-12b-it", "d_in": 3840, "d_sae": 16384},
        "layer": {"engineering_layer": 29, "hook_name": "blocks.29.hook_resid_post"},
    }

    def fake_load(model_path, sae_path, *, device="cuda", dtype="bfloat16",
                  expected_model_revision=None, expected_sae_revision=None):
        return model, object(), "blocks.29.hook_resid_post", provenance

    import sae_concept_lab.extracted_runtime.gemma_loader as gemma_loader_module

    monkeypatch.setattr(gemma_loader_module, "load_gemma_it_target", fake_load)
    return model


def test_gemma_tokenizes_the_TEMPLATED_text_and_disables_bos_prepending(monkeypatch):
    from sae_concept_lab.core.gemma_backend import GemmaRuntimeBackend

    model = _install_gemma_loader(monkeypatch)
    backend = GemmaRuntimeBackend(model_path="/does/not/matter", sae_path="/does/not/matter")
    backend.generate(_baseline_request("hi", history=(("user", "earlier"),)))

    # The template ran, and it saw the history, not just this turn.
    assert model.tokenizer.last_messages is not None
    assert [m["content"] for m in model.tokenizer.last_messages] == ["earlier", "hi"]
    # The raw prompt is NOT what got tokenized -- that is the whole defect.
    assert model.tokenizer.last_add_special_tokens in (None, True)
    # HookedTransformer would otherwise put a second BOS in front of the
    # template's own.
    assert model.last_prepend_bos is False
