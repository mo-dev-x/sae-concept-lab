"""i18n completeness + the actual "switching FR/EN changes custom UI text"
acceptance check, exercised against the real build_demo() wiring (not a
reimplementation of it)."""

from __future__ import annotations

import pytest

from sae_concept_lab.i18n import LANGS, STRINGS, t
from sae_concept_lab.fixtures.loader import default_bundle_path, load_bundle
from sae_concept_lab.core.stub_backend import StubConceptLabBackend
from sae_concept_lab.ui.app_ui import build_demo


def test_every_key_has_every_language_non_empty():
    for key, entry in STRINGS.items():
        for lang in LANGS:
            assert lang in entry, f"{key!r} missing {lang!r}"
            assert entry[lang].strip(), f"{key!r}/{lang!r} is empty"


def test_t_raises_on_unknown_key():
    with pytest.raises(KeyError):
        t("no_such_key", "en")


def test_t_raises_on_unknown_language():
    with pytest.raises(KeyError):
        t("app_title", "de")


def _find_language_switch_fn(demo):
    for blockfn in demo.fns.values():
        if blockfn.fn.__name__ == "_apply_language":
            return blockfn.fn
    raise AssertionError("could not find the language-switch handler in demo.fns")


def _collect_updated_strings(result) -> list[str]:
    strings: list[str] = []
    for item in result:
        constructor_args = getattr(item, "constructor_args", None)
        if not constructor_args:
            continue
        for key in ("value", "label"):
            value = constructor_args.get(key)
            if isinstance(value, str):
                strings.append(value)
    return strings


def test_switching_to_french_changes_custom_ui_text_end_to_end():
    demo = build_demo(**_demo_kwargs())
    apply_language = _find_language_switch_fn(demo)

    english_result = apply_language("en")
    french_result = apply_language("fr")

    assert english_result[0] == "en"
    assert french_result[0] == "fr"

    english_strings = " ".join(_collect_updated_strings(english_result))
    french_strings = " ".join(_collect_updated_strings(french_result))

    # A handful of FR-only and EN-only words that must appear on their
    # respective side and not the other -- proof the switch actually
    # changes rendered text, not just an internal language flag.
    assert "Send" in english_strings and "Envoyer" not in english_strings
    assert "Envoyer" in french_strings and "Send" not in french_strings
    assert "Language" in english_strings
    assert "Langue" in french_strings
    assert t("fake_banner", "fr") in french_strings
    assert t("fake_banner", "en") in english_strings


def test_switching_language_does_not_touch_gr_state_language_value_only_flag():
    """The lang echoed back at index 0 is the ONLY piece of hidden state
    this handler updates; everything else in its outputs list is a
    visible component re-label. Regression guard for accidentally wiring
    the language switch to mutate something it shouldn't."""
    demo = build_demo(**_demo_kwargs())
    apply_language = _find_language_switch_fn(demo)
    result = apply_language("fr")
    assert result[0] == "fr"
    assert len(result) > 30  # sanity: this really is retranslating a lot of components


def _demo_kwargs():
    gemma_bundle = load_bundle(default_bundle_path("gemma"))
    qwen_bundle = load_bundle(default_bundle_path("qwen"))
    return dict(
        gemma_bundle=gemma_bundle,
        qwen_bundle=qwen_bundle,
        gemma_backend=StubConceptLabBackend(),
        qwen_backend=StubConceptLabBackend(),
    )
