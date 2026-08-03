"""Bounded tiny-model overfit and gradient-path diagnostic for Population Language L0."""
from __future__ import annotations

import json
import math
import pathlib
import time
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from . import l0_protocol as protocol
from .l0_data import LanguageBatch
from .l0_models import MatchedCausalTransformer, PopulationLanguageOrganism

VERSION = "population-language-l0-overfit-diagnostic-v0"
BRANCH = "agent/population-language-l0-overfit-diagnostic-v0"
BASE_HEAD = "8da5038f9bb1878c9779bab68fa0f8e2aae88743"
STATUS = "DEVELOPMENT_ONLY_TINY_OVERFIT_DIAGNOSTIC"
PASS = "POPULATION_LANGUAGE_L0_TINY_OVERFIT_PASSES"
FAIL = "POPULATION_LANGUAGE_L0_TINY_OVERFIT_FAILED"
SEED = 120100
STEPS = 256
LEARNING_RATE = 0.003
WORKER_COUNT = 4
COMMUNICATION_ROUNDS = 2
TOP_K = 2

TRANSFORMER_CONFIG = protocol.TransformerConfig(
    d_model=64,
    layers=2,
    feed_forward=128,
    heads=4,
)
POPULATION_CONFIG = protocol.PopulationConfig(
    token_width=64,
    lexical_encoder_width=128,
    worker_width=32,
    worker_feed_forward=64,
    lexical_decoder_width=128,
    router_dim=8,
    training_workers=WORKER_COUNT,
)

_TRANSFORMER_GRADIENT_PATHS = (
    "token_embedding.weight",
    "blocks.0.attention.qkv.weight",
    "blocks.0.feed_forward_input.weight",
    "lm_bias",
)
_POPULATION_GRADIENT_PATHS = (
    "token_embedding.weight",
    "lexical_encoder.0.weight",
    "lexical_encoder.2.weight",
    "initializer.weight",
    "router_query.weight",
    "router_key.weight",
    "message_encoder.weight",
    "gru.input_weight",
    "gru.hidden_weight",
    "worker_ff_input.weight",
    "worker_ff_output.weight",
    "lexical_decoder.0.weight",
    "lexical_decoder.2.weight",
    "lm_bias",
)


def binding_batch(*, device: torch.device | str = "cpu") -> tuple[LanguageBatch, Tensor]:
    """Create four examples sharing the same query but requiring different definitions."""
    pad_id = protocol.TOKEN_TO_ID["<pad>"]
    full = torch.full(
        (4, protocol.MAX_SEQUENCE_LENGTH),
        pad_id,
        dtype=torch.long,
        device=device,
    )
    answer_positions = torch.zeros_like(full, dtype=torch.bool)
    first_answer_positions = torch.zeros_like(full, dtype=torch.bool)

    lhs_nonce = "dax"
    rhs_nonce = "wug"
    relation = "left_of"
    for row in range(4):
        lhs_color = protocol.COLOR_TOKENS[row]
        lhs_shape = protocol.SHAPE_TOKENS[row]
        rhs_color = protocol.COLOR_TOKENS[(row + 3) % len(protocol.COLOR_TOKENS)]
        rhs_shape = protocol.SHAPE_TOKENS[(row + 5) % len(protocol.SHAPE_TOKENS)]
        lhs_definition = (
            "<def>", lhs_nonce, "means", lhs_color, lhs_shape, "<sep>",
        )
        rhs_definition = (
            "<def>", rhs_nonce, "means", rhs_color, rhs_shape, "<sep>",
        )
        definitions = (
            rhs_definition + lhs_definition if row % 2 else lhs_definition + rhs_definition
        )
        query = (
            "<query>", "the", lhs_nonce, "is", relation, "the", rhs_nonce, "<sep>",
        )
        answer = (
            "<answer>", lhs_color, lhs_shape, relation, rhs_color, rhs_shape, "<eos>",
        )
        tokens = ("<bos>",) + definitions + query + answer
        answer_start = len(tokens) - len(answer) + 1
        token_ids = torch.tensor(
            [protocol.TOKEN_TO_ID[token] for token in tokens],
            dtype=torch.long,
            device=device,
        )
        full[row, : token_ids.numel()] = token_ids
        answer_positions[row, answer_start : len(tokens) - 1] = True
        first_answer_positions[row, answer_start] = True

    input_ids = full[:, :-1].contiguous()
    target_ids = full[:, 1:].contiguous()
    loss_mask = target_ids != pad_id
    answer_mask = answer_positions[:, 1:].contiguous()
    first_answer_mask = first_answer_positions[:, 1:].contiguous()
    batch = LanguageBatch(
        input_ids=input_ids,
        target_ids=target_ids,
        loss_mask=loss_mask,
        answer_mask=answer_mask,
        ordinals=(0, 1, 2, 3),
    )
    if not bool(torch.all(answer_mask.sum(dim=1) == 5)):
        raise RuntimeError("tiny binding batch answer span drifted")
    if not bool(torch.all(first_answer_mask.sum(dim=1) == 1)):
        raise RuntimeError("tiny binding batch first-answer span drifted")
    return batch, first_answer_mask


def ablate_definition_values(input_ids: Tensor) -> Tensor:
    if input_ids.dtype != torch.long or input_ids.ndim != 2:
        raise ValueError("definition ablation requires rank-2 torch.long input")
    if input_ids.shape[1] < 12:
        raise ValueError("definition ablation input is too short")
    ablated = input_ids.clone()
    unknown = protocol.TOKEN_TO_ID["<unk>"]
    for position in (4, 5, 10, 11):
        ablated[:, position] = unknown
    return ablated


def masked_loss(logits: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
    if logits.ndim != 3 or targets.shape != mask.shape or logits.shape[:2] != targets.shape:
        raise ValueError("masked language-loss shapes disagree")
    if mask.dtype != torch.bool or not bool(torch.any(mask)):
        raise ValueError("masked language loss requires a nonempty boolean mask")
    return F.cross_entropy(logits[mask], targets[mask])


def _metrics(logits: Tensor, batch: LanguageBatch, first_answer_mask: Tensor) -> dict[str, float]:
    predictions = torch.argmax(logits, dim=-1)
    answer_correct = predictions[batch.answer_mask] == batch.target_ids[batch.answer_mask]
    episode_exact = torch.all(
        (predictions == batch.target_ids) | ~batch.answer_mask,
        dim=1,
    )
    first_correct = (
        predictions[first_answer_mask] == batch.target_ids[first_answer_mask]
    )
    return {
        "answer_loss": float(masked_loss(logits, batch.target_ids, batch.answer_mask).item()),
        "first_answer_loss": float(
            masked_loss(logits, batch.target_ids, first_answer_mask).item()
        ),
        "answer_token_accuracy": float(answer_correct.to(torch.float64).mean().item()),
        "answer_exact_accuracy": float(episode_exact.to(torch.float64).mean().item()),
        "first_answer_accuracy": float(first_correct.to(torch.float64).mean().item()),
    }


def _gradient_report(model: nn.Module, required_paths: tuple[str, ...]) -> dict[str, Any]:
    parameters = dict(model.named_parameters())
    norms: dict[str, float | None] = {}
    valid = True
    for path in required_paths:
        parameter = parameters.get(path)
        if parameter is None or parameter.grad is None:
            norms[path] = None
            valid = False
            continue
        norm = float(torch.linalg.vector_norm(parameter.grad.detach()).item())
        norms[path] = norm
        if not math.isfinite(norm) or norm <= 0.0:
            valid = False
    return {"valid": valid, "norms": norms}


def _train_model(
    name: str,
    model: nn.Module,
    batch: LanguageBatch,
    first_answer_mask: Tensor,
    required_gradient_paths: tuple[str, ...],
    *,
    steps: int,
) -> dict[str, Any]:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=0.0
    )
    checkpoints = {0, 16, 64, 128, steps}
    curves: list[dict[str, float | int]] = []
    started = time.perf_counter()
    gradient_report: dict[str, Any] | None = None

    for step in range(steps + 1):
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch.input_ids)
        loss = masked_loss(logits, batch.target_ids, batch.answer_mask)
        loss.backward()
        if step == 0:
            gradient_report = _gradient_report(model, required_gradient_paths)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step in checkpoints:
            with torch.no_grad():
                evaluated = model(batch.input_ids)
                metrics = _metrics(evaluated, batch, first_answer_mask)
            curves.append({"step": step, **metrics})

    with torch.no_grad():
        clean_logits = model(batch.input_ids)
        ablated_logits = model(ablate_definition_values(batch.input_ids))
        clean = _metrics(clean_logits, batch, first_answer_mask)
        ablated = _metrics(ablated_logits, batch, first_answer_mask)

    if gradient_report is None:
        raise RuntimeError("tiny overfit diagnostic did not inspect gradients")
    passed = (
        gradient_report["valid"]
        and clean["answer_token_accuracy"] == 1.0
        and clean["answer_exact_accuracy"] == 1.0
        and clean["first_answer_accuracy"] == 1.0
        and ablated["first_answer_accuracy"] <= 0.5
        and ablated["first_answer_loss"] >= clean["first_answer_loss"] + 1.0
    )
    return {
        "name": name,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "steps": steps,
        "seconds": time.perf_counter() - started,
        "gradient_report": gradient_report,
        "curves": curves,
        "clean": clean,
        "definition_ablated": ablated,
        "passed": passed,
    }


def classify(rows: list[dict[str, Any]]) -> str:
    if [row.get("name") for row in rows] != ["transformer", "population"]:
        raise ValueError("tiny overfit diagnostic requires ordered transformer/population rows")
    return PASS if all(row.get("passed") is True for row in rows) else FAIL


def run(output_root: pathlib.Path, *, steps: int = STEPS) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"Population Language diagnostic output exists: {output_root}")
    if type(steps) is not int or steps <= 0:
        raise ValueError("tiny overfit steps must be a positive integer")

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cpu")
    batch, first_answer_mask = binding_batch(device=device)

    torch.manual_seed(SEED)
    transformer = MatchedCausalTransformer(TRANSFORMER_CONFIG).to(device)
    transformer_row = _train_model(
        "transformer",
        transformer,
        batch,
        first_answer_mask,
        _TRANSFORMER_GRADIENT_PATHS,
        steps=steps,
    )
    del transformer

    torch.manual_seed(SEED)
    population = PopulationLanguageOrganism(
        POPULATION_CONFIG,
        communication_rounds=COMMUNICATION_ROUNDS,
        top_k=TOP_K,
    ).to(device)
    population_row = _train_model(
        "population",
        population,
        batch,
        first_answer_mask,
        _POPULATION_GRADIENT_PATHS,
        steps=steps,
    )

    rows = [transformer_row, population_row]
    summary = {
        "status": STATUS,
        "version": VERSION,
        "branch": BRANCH,
        "base_head": BASE_HEAD,
        "diagnosis": classify(rows),
        "seed": SEED,
        "steps": steps,
        "learning_rate": LEARNING_RATE,
        "binding_examples": 4,
        "same_query_across_examples": True,
        "rows": rows,
        "boundaries": {
            "tiny_diagnostic_only": True,
            "answer_span_training_used": True,
            "reference_19m_training_performed": False,
            "validation_or_test_data_used": False,
            "population_scaling_claimed": False,
            "natural_language_capability_claimed": False,
            "kv_cache_benchmarked": False,
            "gate9_evidence_modified": False,
        },
    }
    output_root.mkdir(parents=True)
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary
