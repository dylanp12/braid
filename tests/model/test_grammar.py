from dataclasses import replace
from datetime import UTC, datetime, timedelta

from braid.model.grammar import (
    DecodeContext,
    GrammarSchema,
    GraphPatchGrammar,
    PatchEvent,
    RelationRule,
    RoleRule,
    VisibleEvidence,
    VisibleNode,
)

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def context() -> DecodeContext:
    schema = GrammarSchema(
        version="v2",
        type_handles=frozenset({0, 1}),
        relations={0: RelationRule(0, (RoleRule(0, 0), RoleRule(1, 1)))},
    )
    return DecodeContext(
        cutoff=NOW,
        horizon_end=NOW + timedelta(days=7),
        schema=schema,
        nodes={
            0: VisibleNode(0, 0, NOW - timedelta(days=2)),
            1: VisibleNode(1, 1, NOW - timedelta(days=1)),
        },
        evidence={0: VisibleEvidence(0, NOW - timedelta(hours=1))},
        event_handles=frozenset({8}),
    )


def test_valid_assertion_patch() -> None:
    event = PatchEvent(
        event_handle=9,
        operation="ASSERT",
        observed_at=NOW + timedelta(hours=1),
        valid_from=NOW + timedelta(hours=1),
        schema_version="v2",
        relation_handle=0,
        arguments={0: (0,), 1: (1,)},
        basis_refs=(0,),
    )
    assert GraphPatchGrammar().validate_patch(context(), [event]) == ()


def test_grammar_rejects_wrong_type_future_evidence_and_duplicate_event() -> None:
    ctx = context()
    event = PatchEvent(
        event_handle=8,
        operation="ASSERT",
        observed_at=NOW + timedelta(hours=1),
        valid_from=NOW,
        schema_version="v2",
        relation_handle=0,
        arguments={0: (1,), 1: (0,)},
        basis_refs=(999,),
    )
    codes = {item.code for item in GraphPatchGrammar().validate_patch(ctx, [event])}
    assert {"duplicate_event", "evidence_handle", "argument_type"}.issubset(codes)


def test_patch_decode_state_does_not_mutate_source_context() -> None:
    ctx = context()
    create = PatchEvent(
        event_handle=9,
        operation="CREATE_NODE",
        observed_at=NOW + timedelta(hours=1),
        valid_from=NOW,
        schema_version="v2",
        node_handle=2,
        node_type_handle=1,
        payload={"title": "future"},
    )
    expose = PatchEvent(
        event_handle=10,
        operation="EXPOSE",
        observed_at=NOW + timedelta(hours=2),
        valid_from=NOW,
        schema_version="v2",
        node_handle=2,
    )
    assert GraphPatchGrammar().validate_patch(ctx, [create, expose]) == ()
    assert 2 not in ctx.nodes


def test_duplicate_assertion_key_ignores_changed_valid_time() -> None:
    ctx = replace(context(), assertion_keys=frozenset({(0, ((0, (0,)), (1, (1,))))}))
    event = PatchEvent(
        event_handle=9,
        operation="ASSERT",
        observed_at=NOW + timedelta(hours=1),
        valid_from=NOW + timedelta(days=3),
        schema_version="v2",
        relation_handle=0,
        arguments={0: (0,), 1: (1,)},
    )

    codes = {item.code for item in GraphPatchGrammar().validate_patch(ctx, [event])}
    assert "duplicate_assertion" in codes
