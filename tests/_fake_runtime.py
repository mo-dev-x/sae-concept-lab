"""Shared fakes for testing QwenRuntimeBackend/GemmaRuntimeBackend's own
request-translation and wiring logic WITHOUT torch, transformers,
transformer_lens, or sae_lens installed -- this product's base install
deliberately has none of them (see extracted_runtime/__init__.py).

Every fake here replaces a torch-dependent extraction point at the exact
seam the backend itself imports it from (a local import inside generate()),
via monkeypatch.setattr on the real module attribute -- never a patch of
the backend's own code. What is NOT re-tested by using these fakes: the
real numeric behavior of _make_clamp_hook/wrap_hook_with_diagnostics
(masking math, encode/decode round-trip) -- that is covered separately,
when torch is installed, by tests/test_runtime_hooks_differential.py
(pytest.importorskip("torch")-gated). These fakes exist to prove the
BACKEND's own translation (ResolvedControlState -> hook call arguments,
ABLATE->0.0, positions passthrough, diagnostics assembly, same-layer/
multi-SAE/cross-layer enforcement, Compare's baseline arm) is correct,
independent of whether real tensor math is available in this environment.
"""

from __future__ import annotations

import sys
import types


class FakeNoGrad:
    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def install_fake_torch(monkeypatch) -> types.ModuleType:
    fake_torch = types.ModuleType("torch")
    fake_torch.manual_seed = lambda seed: None
    fake_torch.no_grad = lambda: FakeNoGrad()
    fake_torch.bfloat16 = "torch.bfloat16 (fake)"
    fake_torch.float32 = "torch.float32 (fake)"
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    return fake_torch


class FakeTensor:
    """A minimal stand-in for a torch.Tensor: only .shape and slicing on
    the outer (batch) dimension, which is all qwen_backend.py/
    gemma_backend.py's own code touches directly."""

    def __init__(self, nested: list[list[int]]):
        self._nested = nested

    @property
    def shape(self):
        return (len(self._nested), len(self._nested[0]) if self._nested else 0)

    def __getitem__(self, index):
        return self._nested[index]


class FakeInputs(dict):
    def to(self, device):
        return self


class FakeTokenizer:
    """Deterministic, torch-free stand-in for AutoTokenizer/HookedTransformer's
    own .tokenizer: one fake token id per prompt character (never a real
    vocabulary -- this is a translation-logic test, not a tokenization test).

    It publishes a chat_template and apply_chat_template because the REAL
    instruction-tuned tokenizers do. A fake that omitted them would be more
    permissive than reality and would let a backend that never renders a chat
    template pass its tests -- which is exactly how this product shipped
    document-continuation prompts to two -it models while 345 tests were
    green. A fake must not accept what the real thing refuses.
    """

    chat_template = "fake-chat-template"
    #: A fake vocabulary has no real BOS, so there is none to double here.
    #: assert_at_most_one_leading_bos is covered directly, as a pure function,
    #: in tests/test_chat_render.py rather than pretended at through a fake.
    bos_token_id = None

    def __init__(self):
        self.last_apply_kwargs: dict | None = None
        self.last_add_special_tokens: bool | None = None
        self.last_messages: list | None = None

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=False, **kwargs):
        assert tokenize is False, "this product renders to text, then tokenizes separately"
        self.last_messages = list(messages)
        self.last_apply_kwargs = dict(kwargs)
        parts = [f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages]
        if add_generation_prompt:
            parts.append("<assistant>")
        return "".join(parts)

    def __call__(self, prompt, return_tensors="pt", add_special_tokens=True):
        self.last_add_special_tokens = add_special_tokens
        ids = [ord(c) % 50 for c in prompt] or [0]
        return FakeInputs(input_ids=FakeTensor([ids]))

    def decode(self, ids, skip_special_tokens=True):
        return f"generated-tokens:{list(ids)}"


def _resid(batch: int, seq_len: int):
    class _Resid:
        shape = (batch, seq_len, 8)

    return _Resid()


class FakeQwenDecoderLayer:
    """What get_qwen_decoder_layer (real, torch-free) returns:
    text_decoder.layers[layer]. Holds whatever hook gets registered on it,
    so FakeQwenHfModel.generate() can invoke the SAME hook the backend
    attached -- mirroring the real relationship (register_forward_hook on
    a decoder layer, then a forward pass through the model invokes it)."""

    def __init__(self):
        self.hook_fn = None

    def register_forward_hook(self, fn):
        self.hook_fn = fn

        class _Handle:
            def remove(_self):
                self.hook_fn = None

        return _Handle()


class FakeQwenTextDecoder:
    """What resolve_qwen_text_decoder (real, torch-free) would have
    returned: an object with a plain-list .layers attribute, exactly what
    get_qwen_decoder_layer's own `text_decoder.layers[layer]` indexes."""

    def __init__(self, num_layers: int = 1):
        self.layers = [FakeQwenDecoderLayer() for _ in range(num_layers)]


class FakeQwenHfModel:
    """Fake AutoModelForCausalLM stand-in: .generate() invokes whichever
    decoder layer currently holds a registered hook, the exact number of
    times a real prefill+decode loop would (1 prefill call with the full
    prompt shape, then max_new_tokens-1 decode calls with a single-position
    shape) -- so a trace built through the fake diagnostics wrapper
    reflects a genuine call count."""

    def __init__(self, text_decoder: FakeQwenTextDecoder):
        self._text_decoder = text_decoder

    def generate(self, *, input_ids, max_new_tokens=8, do_sample=False):
        prompt_len = input_ids.shape[1]
        active_hooks = [layer.hook_fn for layer in self._text_decoder.layers if layer.hook_fn is not None]
        for hook_fn in active_hooks:
            hook_fn(_resid(1, prompt_len), None)
            for _ in range(max(max_new_tokens - 1, 0)):
                hook_fn(_resid(1, 1), None)
        new_ids = list(range(900, 900 + max_new_tokens))
        return FakeTensor([list(input_ids[0]) + new_ids])


class FakeGemmaModel:
    """Fake HookedTransformer stand-in (Gemma path): .to_tokens(), .generate(),
    .hooks(fwd_hooks=[...]) context manager, .tokenizer.decode()."""

    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self._active_hook = None

    def to_tokens(self, prompt, prepend_bos=True):
        # Recorded so a test can assert the backend disables prepending on
        # already-templated text; the real HookedTransformer would otherwise
        # add a second BOS on top of the template's own.
        self.last_prepend_bos = prepend_bos
        return self.tokenizer(prompt)["input_ids"]

    def hooks(self, fwd_hooks):
        model = self

        class _Ctx:
            def __enter__(_self):
                model._active_hook = fwd_hooks[0][1]
                return model

            def __exit__(_self, *exc_info):
                model._active_hook = None
                return False

        return _Ctx()

    def generate(self, tokens, max_new_tokens=8, do_sample=False, verbose=False):
        prompt_len = tokens.shape[1]
        if self._active_hook is not None:
            self._active_hook(_resid(1, prompt_len), None)
            for _ in range(max(max_new_tokens - 1, 0)):
                self._active_hook(_resid(1, 1), None)
        new_ids = list(range(900, 900 + max_new_tokens))
        return FakeTensor([list(tokens[0]) + new_ids])


def install_fake_qwen_transformers(monkeypatch) -> None:
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda path: FakeTokenizer())
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)


def fake_make_clamp_hook(sae_fp32, feature_index, clamp_value, positions, prompt_lengths, stats):
    """Pure-Python stand-in for the real _make_clamp_hook: does no tensor
    math at all, just returns resid unchanged. Callers assert on the
    ARGUMENTS this was invoked with (via a mutable capture list), not on
    numeric behavior -- the real numeric contract is covered by
    test_runtime_hooks_differential.py when torch is installed."""

    def hook_fn(resid, hook):
        return resid

    return hook_fn


def make_fake_wrap_hook_with_diagnostics():
    """Returns (fake_wrap_fn, calls) -- calls records every invocation's
    kwargs, and fake_wrap_fn builds a REAL InterventionTrace (imported here,
    torch-free) into trace_out on every hook call, alternating prefill/
    decode by call index -- exactly the shape mechanical_verdict/
    find_first_disappearance_boundary (both real, torch-free, unfaked)
    expect, so those two functions can run for real against these fake
    traces."""
    from sae_concept_lab.extracted_runtime.diagnostics import InterventionTrace

    calls: list[dict] = []

    def fake_wrap(
        inner_hook_fn,
        *,
        sae,
        feature_index,
        mode,
        dose_or_raw_label,
        calibration_input,
        resolved_absolute_target,
        hook_name,
        trace_out,
    ):
        calls.append(
            {
                "feature_index": feature_index,
                "mode": mode,
                "resolved_absolute_target": resolved_absolute_target,
                "hook_name": hook_name,
            }
        )
        counter = {"n": 0}

        def hook_fn(resid, hook):
            index = counter["n"]
            counter["n"] += 1
            classification = "prefill" if index == 0 else "decode"
            # Deliberately NOT masking-aware: this fake proves the BACKEND's
            # own wiring (arguments passed to _make_clamp_hook/this wrapper,
            # diagnostics assembled from whatever trace comes back), not the
            # real masking numerics -- see test_runtime_hooks_differential.py
            # for the real, torch-gated, masking-aware behavior.
            delta_norm = abs(resolved_absolute_target)
            trace_out.append(
                InterventionTrace(
                    call_index=index,
                    call_classification=classification,
                    requested_mode=mode,
                    requested_dose_or_raw=dose_or_raw_label,
                    calibration_input=calibration_input,
                    resolved_absolute_target=resolved_absolute_target,
                    backend_received_value=resolved_absolute_target,
                    hook_name=hook_name,
                    hooked_tensor_shape=tuple(resid.shape),
                    feature_activation_before=0.0,
                    assigned_feature_value=resolved_absolute_target,
                    feature_activation_after=resolved_absolute_target,
                    residual_delta_norm=delta_norm,
                    residual_norm=1.0,
                )
            )
            return inner_hook_fn(resid, hook)

        return hook_fn

    return fake_wrap, calls


def fake_register_qwen_raw_hook(decoder_layer_module, hook_fn):
    return decoder_layer_module.register_forward_hook(hook_fn)
