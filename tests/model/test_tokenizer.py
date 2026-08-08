from dataclasses import replace

from braid.contract.types import (
    NodeTypeDecl,
    RelationDecl,
    RoleDecl,
    SchemaDecl,
    SupportExample,
)
from braid.model.tokenizer import BraidTokenizer, DynamicHandleTable, HandleKind


def test_byte_fallback_round_trip_and_is_deterministic() -> None:
    tokenizer = BraidTokenizer()
    text = "naïve graph 🧶"
    assert tokenizer.decode_text(tokenizer.encode_text(text)) == text
    assert tokenizer.encode_text(text) == tokenizer.encode_text(text)


def test_external_renaming_leaves_schema_model_view_unchanged() -> None:
    tokenizer = BraidTokenizer()
    first_types = [
        {"id": "person", "description": "a person"},
        {"id": "project", "description": "a project"},
    ]
    first_relations = [
        {
            "id": "owns",
            "description": "ownership",
            "roles": [
                {"id": "owner", "target_type": "person", "min_count": 1, "max_count": 1},
                {"id": "thing", "target_type": "project", "min_count": 1, "max_count": 1},
            ],
        }
    ]
    renamed_types = [
        {"id": "x7", "description": "a person"},
        {"id": "x9", "description": "a project"},
    ]
    renamed_relations = [
        {
            "id": "r4",
            "description": "ownership",
            "roles": [
                {"id": "q1", "target_type": "x7", "min_count": 1, "max_count": 1},
                {"id": "q2", "target_type": "x9", "min_count": 1, "max_count": 1},
            ],
        }
    ]

    handles_a, items_a = tokenizer.encode_schema(types=first_types, relations=first_relations)
    handles_b, items_b = tokenizer.encode_schema(types=renamed_types, relations=renamed_relations)

    assert handles_a.model_view() == handles_b.model_view()
    assert [(item.kind, item.handle, item.token_ids) for item in items_a] == [
        (item.kind, item.handle, item.token_ids) for item in items_b
    ]


def test_dynamic_handles_use_observation_order_not_identifier_sorting() -> None:
    table = DynamicHandleTable()
    assert table.register(HandleKind.NODE, "z").index == 0
    assert table.register(HandleKind.NODE, "a").index == 1
    assert table.register(HandleKind.NODE, "z").index == 0


def test_schema_constraints_inverses_and_support_examples_are_rename_equivariant() -> None:
    tokenizer = BraidTokenizer()
    first = _rich_schema(
        person="person",
        project="project",
        owns="owns",
        owned_by="owned_by",
        owner="owner",
        thing="thing",
    )
    renamed = _rich_schema(
        person="t7",
        project="t9",
        owns="r4",
        owned_by="r8",
        owner="q1",
        thing="q2",
    )

    handles_a, items_a = tokenizer.encode_schema_decl(first)
    handles_b, items_b = tokenizer.encode_schema_decl(renamed)

    assert handles_a.model_view() == handles_b.model_view()
    assert [(item.kind, item.handle, item.token_ids) for item in items_a] == [
        (item.kind, item.handle, item.token_ids) for item in items_b
    ]

    without_support = replace(first, support_examples=())
    without_schema_constraints = replace(first, constraints={})
    without_inverse = replace(
        first,
        relations=tuple(replace(relation, inverse=None) for relation in first.relations),
    )
    encoded = tuple(item.token_ids for item in items_a)
    assert encoded != tuple(
        item.token_ids for item in tokenizer.encode_schema_decl(without_support)[1]
    )
    assert encoded != tuple(
        item.token_ids for item in tokenizer.encode_schema_decl(without_schema_constraints)[1]
    )
    assert encoded != tuple(
        item.token_ids for item in tokenizer.encode_schema_decl(without_inverse)[1]
    )


def _rich_schema(
    *,
    person: str,
    project: str,
    owns: str,
    owned_by: str,
    owner: str,
    thing: str,
) -> SchemaDecl:
    return SchemaDecl(
        version="v2",
        observed_at=None,
        node_types=(NodeTypeDecl(person, "a person"), NodeTypeDecl(project, "a project")),
        relations=(
            RelationDecl(
                owns,
                (RoleDecl(owner, (person,)), RoleDecl(thing, (project,))),
                "ownership",
                inverse=owned_by,
            ),
            RelationDecl(
                owned_by,
                (RoleDecl(thing, (project,)), RoleDecl(owner, (person,))),
                "inverse ownership",
                inverse=owns,
            ),
        ),
        constraints={"primary_relation": owns, "actor_type": person},
        support_examples=(
            SupportExample(
                relation=owns,
                role_types={owner: person, thing: project},
                text="a structural ownership example",
            ),
        ),
    )
