"""Safe, constrained inference at the public ForecastDistribution boundary."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from datetime import timedelta

import torch

from braid.contract.types import (
    DerivationRecord,
    EventMarginal,
    ForecastDistribution,
    ForecastRequest,
    GraphEventV2,
    PatchWindow,
    ProvenanceRecord,
    RoleBinding,
)
from braid.contract.types import Operation as ContractOperation
from braid.contract.validate import validate_forecast_distribution, validate_forecast_request
from braid.model.checkpoint import (
    CheckpointCompatibilityError,
    CheckpointMetadata,
    checkpoint_is_bound,
)
from braid.model.grammar import GraphPatchGrammar, Operation, PatchEvent
from braid.model.model import EventGraphModel
from braid.model.retrieval import RetrievalResult, TypedRetriever
from braid.model.tensorize import TensorizedForecast, tensorize_forecast_request
from braid.model.tokenizer import BraidTokenizer, HandleKind


@dataclass(frozen=True, slots=True)
class CalibrationRecord:
    """An explicit temperature fitted on the checkpoint-bound evaluation artifact."""

    temperature: float
    fitted_on_hash: str
    method: str = "temperature-scaling"

    def __post_init__(self) -> None:
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("calibration temperature must be positive and finite")
        if len(self.fitted_on_hash) != 64 or any(
            value not in "0123456789abcdef" for value in self.fitted_on_hash
        ):
            raise ValueError("fitted_on_hash must be a lowercase SHA-256 digest")
        if not self.method:
            raise ValueError("calibration method cannot be empty")


class EventGraphForecaster:
    """Turn a trained, calibrated model into constrained proposal distributions."""

    def __init__(
        self,
        model: EventGraphModel,
        tokenizer: BraidTokenizer,
        checkpoint: CheckpointMetadata,
        *,
        calibration: CalibrationRecord | None,
        retriever: TypedRetriever | None = None,
        minimum_retrieval_coverage: float = 0.99,
        require_claim_grade_retrieval: bool = False,
    ) -> None:
        if checkpoint.architecture_hash != model.architecture_fingerprint:
            raise CheckpointCompatibilityError("model and checkpoint architecture hashes differ")
        if checkpoint.tokenizer_hash != tokenizer.fingerprint:
            raise CheckpointCompatibilityError("tokenizer and checkpoint hashes differ")
        if not 0.0 <= minimum_retrieval_coverage <= 1.0:
            raise ValueError("minimum retrieval coverage must be in [0, 1]")
        self.model = model
        self.tokenizer = tokenizer
        self.checkpoint = checkpoint
        self.calibration = calibration
        self.retriever = retriever or TypedRetriever()
        self.minimum_retrieval_coverage = minimum_retrieval_coverage
        self.require_claim_grade_retrieval = require_claim_grade_retrieval

    @property
    def manifest_id(self) -> str:
        return self.checkpoint.to_contract_manifest().manifest_id

    def forecast(self, request: ForecastRequest) -> ForecastDistribution:
        """Return one constrained support window or a machine-readable abstention."""

        validate_forecast_request(request)
        if self.checkpoint.training_steps <= 0:
            return self._abstain("UNTRAINED_MODEL", coverage=0.0)
        if not checkpoint_is_bound(self.model, self.checkpoint):
            return self._abstain("UNVERIFIED_CHECKPOINT", coverage=0.0)
        if self.calibration is None:
            return self._abstain("UNCALIBRATED_MODEL", coverage=0.0)
        if self.calibration.fitted_on_hash != self.checkpoint.evaluation_hash:
            return self._abstain("CALIBRATION_LINEAGE_MISMATCH", coverage=0.0)
        if not request.schema.relations:
            return self._abstain("NO_DECODABLE_RELATIONS", coverage=0.0)
        device = next(self.model.parameters()).device
        episode = tensorize_forecast_request(
            request,
            self.tokenizer,
            time_unit_seconds=self.model.config.time_unit_seconds,
            device=device,
        )
        retrieval = self._retrieve(request, episode)
        if not retrieval.safe_to_generate(self.minimum_retrieval_coverage):
            return self._abstain("INSUFFICIENT_RETRIEVAL_COVERAGE", retrieval.coverage)
        if self.require_claim_grade_retrieval and not retrieval.claim_grade:
            return self._abstain("RETRIEVAL_RECALL_NOT_CLAIM_GRADE", retrieval.coverage)
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                output = self.model(episode.schema_batch, episode.event_batch)
        finally:
            self.model.train(was_training)
        decoded = self._decode_one(request, episode, output.distribution, retrieval)
        if decoded is None:
            return self._abstain("NO_GRAMMAR_VALID_PATCH", retrieval.coverage)
        event, probability, uncertainty = decoded
        result = ForecastDistribution(
            sampled_patch_windows=(PatchWindow((event,), probability),),
            event_marginals=(EventMarginal(event, probability),),
            calibrated_uncertainty=uncertainty,
            abstention_reason=None,
            retrieval_coverage=min(retrieval.coverage, retrieval.type_coverage),
            model_manifest_id=self.manifest_id,
        )
        validate_forecast_distribution(result, request=request)
        return result

    def _retrieve(self, request: ForecastRequest, episode: TensorizedForecast) -> RetrievalResult:
        target_relations = set(request.task.target_relations) or {
            relation.name for relation in request.schema.relations
        }
        required_types = {
            episode.handles.resolve(HandleKind.TYPE, node_type).index
            for relation in request.schema.relations
            if relation.name in target_relations
            for role in relation.roles
            for node_type in role.allowed_node_types
        }
        query_handles = [
            episode.handles.resolve(HandleKind.NODE, node_id).index
            for node_id in request.task.query_node_ids
        ]
        assert request.cutoff is not None
        return self.retriever.retrieve(
            episode.retrieval_nodes,
            cutoff=request.cutoff,
            required_types=required_types,
            query_handles=query_handles,
        )

    def _decode_one(
        self,
        request: ForecastRequest,
        episode: TensorizedForecast,
        distribution: object,
        retrieval: RetrievalResult,
    ) -> tuple[GraphEventV2, float, float] | None:
        assert self.calibration is not None and request.cutoff is not None
        final = int(distribution.event_mask[0].sum().item()) - 1
        operation_logits = distribution.operation_logits[0, final] / self.calibration.temperature
        operation_probabilities = torch.softmax(operation_logits, dim=-1)
        entropy = -(operation_probabilities * operation_probabilities.clamp_min(1e-12).log()).sum()
        uncertainty = float(
            (entropy / math.log(operation_probabilities.numel())).clamp(0, 1).item()
        )
        relation_probabilities = torch.softmax(
            distribution.relation_logits[0, final] / self.calibration.temperature, dim=-1
        )
        retrieved_node_handles = frozenset(candidate.handle for candidate in retrieval.candidates)
        node_probabilities = None
        if distribution.argument_logits.shape[-1] and retrieved_node_handles:
            candidate_count = distribution.argument_logits.shape[-1]
            if any(handle < 0 or handle >= candidate_count for handle in retrieved_node_handles):
                raise ValueError("retriever returned an out-of-range node handle")
            argument_logits = distribution.argument_logits[0, final] / self.calibration.temperature
            retrieved_mask = torch.zeros_like(argument_logits, dtype=torch.bool)
            retrieved_mask[list(sorted(retrieved_node_handles))] = True
            argument_logits = argument_logits.masked_fill(
                ~retrieved_mask, torch.finfo(argument_logits.dtype).min
            )
            node_probabilities = torch.softmax(argument_logits, dim=-1)
        evidence_probabilities = (
            torch.softmax(
                distribution.evidence_logits[0, final] / self.calibration.temperature,
                dim=-1,
            )
            if episode.evidence_ids
            else None
        )
        delta_seconds = min(
            request.horizon.total_seconds(),
            max(
                1e-6,
                math.exp(-float(distribution.delta_log_rate[0, final].item()))
                * self.model.config.time_unit_seconds,
            ),
        )
        observed_at = request.cutoff + timedelta(seconds=delta_seconds)
        valid_from = observed_at + timedelta(
            seconds=float(distribution.valid_lag_mean[0, final].item())
            * self.model.config.time_unit_seconds
        )
        basis_handles: tuple[int, ...] = ()
        basis_ids: tuple[str, ...] = ()
        evidence_probability = 1.0
        if evidence_probabilities is not None:
            selected_evidence = int(torch.argmax(evidence_probabilities).item())
            basis_handles = (selected_evidence,)
            basis_ids = (episode.evidence_ids[selected_evidence],)
            evidence_probability = float(evidence_probabilities[selected_evidence].item())

        allowed_relations = set(request.task.target_relations) or set(episode.relation_names)
        operation_indices = torch.argsort(operation_probabilities, descending=True).tolist()
        for operation_index in operation_indices:
            operation = tuple(Operation)[operation_index]
            built = self._build_candidate(
                request,
                episode,
                operation,
                relation_probabilities,
                node_probabilities,
                observed_at,
                valid_from,
                basis_handles,
                basis_ids,
                allowed_relations,
                retrieved_node_handles,
            )
            if built is None:
                continue
            internal, event, structural_probability = built
            if GraphPatchGrammar().validate_patch(episode.grammar_context, (internal,)):
                continue
            probability = float(
                (
                    operation_probabilities[operation_index]
                    * structural_probability
                    * evidence_probability
                )
                .clamp(0, 1)
                .item()
            )
            return event, probability, uncertainty
        return None

    def _build_candidate(
        self,
        request: ForecastRequest,
        episode: TensorizedForecast,
        operation: Operation,
        relation_probabilities: torch.Tensor,
        node_probabilities: torch.Tensor | None,
        observed_at: object,
        valid_from: object,
        basis_handles: tuple[int, ...],
        basis_ids: tuple[str, ...],
        allowed_relations: set[str],
        retrieved_node_handles: frozenset[int],
    ) -> tuple[PatchEvent, GraphEventV2, torch.Tensor] | None:
        assert request.cutoff is not None
        event_handle = episode.handles.count(HandleKind.EVENT)
        token = hashlib.sha256(
            f"{request.request_id}:{self.manifest_id}:{event_handle}".encode()
        ).hexdigest()[:20]
        event_id = f"proposal:{token}"
        relation_name: str | None = None
        relation_handle: int | None = None
        internal_arguments: dict[int, tuple[int, ...]] = {}
        bindings: list[RoleBinding] = []
        payload: dict[str, object] = {}
        node_handle: int | None = None
        node_type_handle: int | None = None
        target_event_handle: int | None = None
        probability = relation_probabilities.new_tensor(1.0)

        if operation in (Operation.ASSERT, Operation.SUPERSEDE):
            if node_probabilities is None:
                return None
            relation_order = torch.argsort(relation_probabilities, descending=True).tolist()
            selected = next(
                (
                    index
                    for index in relation_order
                    if episode.relation_names[index] in allowed_relations
                ),
                None,
            )
            if selected is None:
                return None
            relation_name = episode.relation_names[selected]
            relation_handle = selected
            probability = probability * relation_probabilities[selected]
            declaration = request.schema.relations[selected]
            for role in declaration.roles:
                role_handle = episode.handles.resolve(
                    HandleKind.ROLE, f"{relation_name}:{role.name}"
                ).index
                allowed_types = {
                    episode.handles.resolve(HandleKind.TYPE, item).index
                    for item in role.allowed_node_types
                }
                candidates = [
                    index
                    for index in torch.argsort(node_probabilities, descending=True).tolist()
                    if index in retrieved_node_handles
                    and episode.node_type_handles[index] in allowed_types
                ]
                if len(candidates) < role.min_count:
                    return None
                chosen = tuple(candidates[: role.min_count])
                internal_arguments[role_handle] = chosen
                bindings.extend(RoleBinding(role.name, episode.node_ids[index]) for index in chosen)
                for index in chosen:
                    probability = probability * node_probabilities[index]
            if operation is Operation.SUPERSEDE:
                target = _latest_assertion(request)
                if target is None:
                    return None
                target_event_handle = episode.handles.resolve(
                    HandleKind.EVENT, target.event_id
                ).index
                payload["target_event_id"] = target.event_id
        elif operation is Operation.RETRACT:
            target = _latest_assertion(request)
            if target is None:
                return None
            target_event_handle = episode.handles.resolve(HandleKind.EVENT, target.event_id).index
            relation_name = target.relation
            relation_handle = (
                None
                if relation_name is None
                else episode.handles.resolve(HandleKind.RELATION, relation_name).index
            )
            bindings = list(target.arguments)
            payload["target_event_id"] = target.event_id
        elif operation is Operation.CREATE_NODE:
            node_handle = len(episode.node_ids)
            node_type_handle = 0
            proposed_node = f"proposal-node:{token}"
            bindings = [RoleBinding("node", proposed_node)]
            payload = {
                "node_id": proposed_node,
                "node_type": request.schema.node_types[node_type_handle].name,
            }
        elif operation is Operation.EXPOSE:
            if node_probabilities is None or not retrieved_node_handles:
                return None
            node_handle = max(
                retrieved_node_handles,
                key=lambda handle: float(node_probabilities[handle].item()),
            )
            probability = probability * node_probabilities[node_handle]
            bindings = [RoleBinding("candidate", episode.node_ids[node_handle])]
        else:
            # UPDATE/SCHEMA_CHANGE need autoregressive structured payloads.  JUDGE is
            # disabled until complete-slate labels exist.  Failing closed is part of
            # the public inference contract.
            return None

        internal = PatchEvent(
            event_handle=event_handle,
            operation=operation,
            observed_at=observed_at,
            valid_from=valid_from,
            schema_version=request.schema.version,
            relation_handle=relation_handle,
            node_handle=node_handle,
            node_type_handle=node_type_handle,
            arguments=internal_arguments,
            payload=payload,
            basis_refs=basis_handles,
            target_event_handle=target_event_handle,
        )
        event = GraphEventV2(
            event_id=event_id,
            observed_at=observed_at,
            valid_from=valid_from,
            valid_to=None,
            operation=ContractOperation(operation.value),
            schema_version=request.schema.version,
            relation=relation_name,
            arguments=tuple(bindings),
            payload=payload,
            basis_refs=basis_ids,
            derivation=DerivationRecord(
                method="braid-eventgraph-v2",
                input_refs=basis_ids,
                parameters={"model_manifest_id": self.manifest_id},
            ),
            provenance=ProvenanceRecord(
                source="braid-model-proposal",
                source_record_id=None,
                license=None,
                acquired_at=observed_at,
            ),
        )
        return internal, event, probability

    def _abstain(self, reason: str, coverage: float) -> ForecastDistribution:
        result = ForecastDistribution(
            sampled_patch_windows=(),
            event_marginals=(),
            calibrated_uncertainty=1.0,
            abstention_reason=reason,
            retrieval_coverage=max(0.0, min(1.0, coverage)),
            model_manifest_id=self.manifest_id,
        )
        validate_forecast_distribution(result)
        return result


def _latest_assertion(request: ForecastRequest) -> GraphEventV2 | None:
    assertions = [
        event
        for event in request.prefix.events
        if event.operation is ContractOperation.ASSERT and event.relation is not None
    ]
    return max(assertions, key=lambda event: event.observed_at) if assertions else None
