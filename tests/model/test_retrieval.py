from datetime import UTC, datetime, timedelta

import pytest

from braid.model.retrieval import NodeCandidate, RetrievalResult, TypedRetriever


def test_retrieval_excludes_future_nodes_and_reports_recall() -> None:
    cutoff = datetime(2026, 1, 1, tzinfo=UTC)
    nodes = [
        NodeCandidate(0, 3, cutoff - timedelta(days=1), semantic_score=0.1),
        NodeCandidate(1, 3, cutoff + timedelta(days=1), semantic_score=1.0),
    ]
    result = TypedRetriever(max_candidates=4).retrieve(
        nodes,
        cutoff=cutoff,
        required_types=[3],
        query_handles=[0],
        true_target_handles=[0, 1],
    )
    assert [item.handle for item in result.candidates] == [0]
    assert result.true_target_recall == 0.5
    assert not result.claim_grade


def test_missing_query_forces_safe_abstention() -> None:
    cutoff = datetime(2026, 1, 1, tzinfo=UTC)
    result = TypedRetriever().retrieve(
        [NodeCandidate(0, 3, cutoff)],
        cutoff=cutoff,
        required_types=[3],
        query_handles=[99],
    )
    assert not result.safe_to_generate()


def test_claim_grade_requires_structural_coverage_and_measured_target_recall() -> None:
    cutoff = datetime(2026, 1, 1, tzinfo=UTC)
    candidate = NodeCandidate(0, 3, cutoff)
    unmeasured = RetrievalResult((candidate,), 1.0, 1.0, None)
    weak_recall = RetrievalResult((candidate,), 1.0, 1.0, 0.98)
    claim_grade = RetrievalResult((candidate,), 1.0, 1.0, 0.99)

    assert unmeasured.safe_to_generate()
    assert not unmeasured.safe_to_generate(require_claim_grade=True)
    assert not weak_recall.claim_grade
    assert claim_grade.claim_grade
    assert claim_grade.safe_to_generate(require_claim_grade=True)


def test_explicit_query_is_retrieved_even_when_not_an_argument_type() -> None:
    cutoff = datetime(2026, 1, 1, tzinfo=UTC)
    result = TypedRetriever(max_candidates=2).retrieve(
        (NodeCandidate(0, 9, cutoff), NodeCandidate(1, 3, cutoff)),
        cutoff=cutoff,
        required_types=(3,),
        query_handles=(0,),
    )

    assert {candidate.handle for candidate in result.candidates} == {0, 1}
    assert result.coverage == 1.0
    assert result.type_coverage == 1.0


def test_candidate_cap_reports_queries_it_could_not_return() -> None:
    cutoff = datetime(2026, 1, 1, tzinfo=UTC)
    result = TypedRetriever(max_candidates=1).retrieve(
        (NodeCandidate(0, 3, cutoff), NodeCandidate(1, 3, cutoff)),
        cutoff=cutoff,
        required_types=(3,),
        query_handles=(0, 1),
    )

    assert result.coverage == 0.5
    assert len(result.missing_query_handles) == 1
    assert not result.safe_to_generate()


def test_retrieval_result_rejects_invalid_coverage() -> None:
    with pytest.raises(ValueError, match="coverage"):
        RetrievalResult((), 1.01, 1.0, None)
