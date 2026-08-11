"""The core reset rule: any change to concept/direction/strength
unconditionally clears the conversation and surfaces a localized notice.
There is no override -- the prior Advanced-only "continue anyway" escape
hatch was removed per the P0 release-safety correction, specifically
because it let Advanced retain stale history across a settings change
while Public could not, making Advanced a second, divergent intervention
system in practice."""

from __future__ import annotations

from sae_concept_lab.core.logic import Selection, apply_selection_change

HISTORY = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "[FAKE STUB] hello"}]


def test_first_selection_ever_does_not_reset():
    result = apply_selection_change(
        previous_selection=None,
        new_selection=Selection("c1", "amplify", "low"),
        history=HISTORY,
    )
    assert result.reset_happened is False
    assert result.new_history == HISTORY
    assert result.notice_key is None


def test_unchanged_selection_does_not_reset():
    selection = Selection("c1", "amplify", "low")
    result = apply_selection_change(
        previous_selection=selection, new_selection=selection, history=HISTORY
    )
    assert result.reset_happened is False
    assert result.new_history == HISTORY
    assert result.notice_key is None


def test_concept_change_resets_and_notifies():
    result = apply_selection_change(
        previous_selection=Selection("c1", "amplify", "low"),
        new_selection=Selection("c2", "amplify", "low"),
        history=HISTORY,
    )
    assert result.reset_happened is True
    assert result.new_history == []
    assert result.notice_key == "reset_notice"


def test_direction_change_resets_and_notifies():
    result = apply_selection_change(
        previous_selection=Selection("c1", "amplify", "low"),
        new_selection=Selection("c1", "suppress", "low"),
        history=HISTORY,
    )
    assert result.reset_happened is True
    assert result.new_history == []
    assert result.notice_key == "reset_notice"


def test_strength_change_resets_and_notifies():
    result = apply_selection_change(
        previous_selection=Selection("c1", "amplify", "low"),
        new_selection=Selection("c1", "amplify", "high"),
        history=HISTORY,
    )
    assert result.reset_happened is True
    assert result.new_history == []
    assert result.notice_key == "reset_notice"


def test_apply_selection_change_has_no_override_parameter():
    """Regression guard for the P0 release-safety correction: there must be
    no way to call apply_selection_change() and suppress a genuine reset --
    the old continue_anyway kwarg must be gone from the signature entirely,
    not merely defaulted to False somewhere."""
    import inspect

    params = inspect.signature(apply_selection_change).parameters
    assert "continue_anyway" not in params


def test_all_three_setting_changes_covered_by_a_single_parametrized_style_check():
    """Explicit acceptance check: 'concept, direction, and strength changes
    unconditionally clear conversation history' -- concept, direction, AND
    strength, each checked independently rather than assumed from one."""
    base = Selection("c1", "amplify", "low")
    variants = [
        Selection("c2", "amplify", "low"),
        Selection("c1", "suppress", "low"),
        Selection("c1", "amplify", "high"),
    ]
    for changed in variants:
        result = apply_selection_change(previous_selection=base, new_selection=changed, history=HISTORY)
        assert result.reset_happened is True, f"expected reset for {changed}"
        assert result.notice_key == "reset_notice"
