from __future__ import annotations

import hashlib
import json
import pathlib


def _verify_gate9d_stage1_seed1_result_when_present() -> None:
    result_root = (
        pathlib.Path(__file__).resolve().parents[1]
        / "experiments"
        / "population_compute_scaling_v0"
    )
    result_path = result_root / (
        "gate9_contextual_failure_decomposition_stage1_seed1_result_v0.json"
    )
    manifest_path = result_root / (
        "gate9_contextual_failure_decomposition_stage1_seed1_source_manifest_v0.sha256"
    )
    if not result_path.exists() and not manifest_path.exists():
        return

    assert result_path.is_file()
    assert manifest_path.is_file()
    result_bytes = result_path.read_bytes()
    assert hashlib.sha256(result_bytes).hexdigest() == (
        "80673d996b5075ff09577c5a6fec412c04ff8c9914c932914c4714674f74d5b5"
    )
    result = json.loads(result_bytes)
    assert result["experiment_version"] == (
        "gate9-contextual-failure-decomposition-stage1-seed1-result-v0"
    )
    assert result["scientific_status"] == "G9D_STAGE1_SEED1_FAILURE_RECORDED"
    assert result["seed_index"] == 1
    assert result["seed_outcome"] == "G9D_STAGE1_SEED_FAILED"
    assert result["role"] == "second_ordered_seed_replication_result_only"
    assert result["stack_identity"] == {
        "architecture_head": "c689cc3f38f6f6f642916ee1a702d7de7bd0e43b",
        "execution_head": "2e1b91d578e7bf9b4c54aa2ee1c120a9ec01b21c",
        "operator_contract_head": "be6451e1af82b18749bd0313a9c02ca62c4eee5c",
        "protocol_head": "8deca15aef78d8636b07570aff044f9b7ae31928",
        "seed0_result_head": "9a36b79453e794886c00c85df67bf0bd8fad7345",
    }
    assert result["operator"] == {
        "counter": 72_057_594_037_927_936,
        "dataset_sha256": "37b4aafb3e184eaa2e8096b649457134bd5025f297912c26ed34100b76a3ff0f",
        "key": 14_550_454_351_299_327_585,
        "non_support_query_count": 247,
    }
    assert result["training"] == {
        "batch_size": 247,
        "examples_seen": 252_928,
        "final_loss": 0.36665672063827515,
        "minimum_loss": 0.36665672063827515,
        "minimum_loss_step": 1_024,
        "rows": 1_024,
        "steps": 1_024,
        "unique_examples": 247,
    }
    assert result["evaluation"] == {
        "bit_accuracy": 0.8598178137651822,
        "bit_correct": 1_699,
        "bit_total": 1_976,
        "exact_accuracy": 0.23481781376518218,
        "full_correct": 58,
        "oracle_accuracy": 1.0,
        "oracle_correct": 247,
        "query_only_accuracy": 0.004048582995951417,
        "query_only_correct": 1,
        "rows": 247,
        "stage_passes": False,
    }
    assert result["diagnostic_consequence"] == {
        "all_three_initialization_seeds_required_to_advance": True,
        "remaining_seed_role": "replication_and_terminal_classification_only",
        "stage1_seed0_independently_failed": True,
        "stage1_seed1_independently_failed": True,
        "stage2_advancement_allowed": False,
        "terminal_outcomes_still_possible": [
            "G9D_BASIC_QUERY_MAPPING_FAILED",
            "G9D_DIAGNOSTIC_INCONCLUSIVE",
        ],
    }
    assert result["closed_boundaries"] == {
        "checkpoint_selection_performed": False,
        "diagnostic_classification_performed": False,
        "gate9_v0_result_mutation_performed": False,
        "later_diagnostic_stage_execution_performed": False,
        "population_execution_performed": False,
        "retraining_performed": False,
        "scientific_execution_performed": False,
        "scientific_test_generation_performed": False,
        "training_performed_in_result_slice": False,
    }
    assert result["independent_audit"] == {
        "checkpoint_all_finite_float32": True,
        "checkpoint_loaded_weights_only": True,
        "checkpoint_parameter_count_verified": True,
        "checkpoint_schema_verified": True,
        "evaluation_ledger_reconstructed": True,
        "git_head_verified": True,
        "git_status_empty_hash_verified_from_terminal": True,
        "manifest_paths_sorted_and_unique": True,
        "manifest_sha256_verified": True,
        "run_config_verified": True,
        "source_artifact_modified": False,
        "summary_evidence_reconciled": True,
        "training_ledger_reconstructed": True,
        "uploaded_artifact_hashes_match_manifest": True,
    }
    assert result["source_artifact_sha256"] == {
        "git-head.txt": "39aeac89e593584f77d1365c779f2eb2e5317e07729124026158f64c89abd940",
        "git-status.txt": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "manifest.sha256": "1694a3a0c23e71b4c830d432234237498666c832b579ce62f2622b007b70fe5f",
        "run-config.json": "6af90abd9443549df614c2c685dcc0bd61648a7478bf5d9745a3790c5d7814f3",
        "seed-1/evaluation-per-episode.jsonl": "418dc7c21f3641baea080d3bdf66103ae8548c929a0d105305120615ecbdc512",
        "seed-1/selected-checkpoint.pt": "db7e0189b8d900a71e0410c229241befab2e0d28a81f51accb0bb32c2195a555",
        "seed-1/summary.json": "b593c53b1d7d1d6de031ec3e04ea65c1a6ce63121a01acfc21d8fe80866aaa06",
        "seed-1/train-steps.jsonl": "95c0a36f0229b030a84be2e7fb3bc9a2f09a32e55d7827904bea6e46a137e5cf",
    }
    expected_manifest = (
        "39aeac89e593584f77d1365c779f2eb2e5317e07729124026158f64c89abd940  git-head.txt\n"
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  git-status.txt\n"
        "6af90abd9443549df614c2c685dcc0bd61648a7478bf5d9745a3790c5d7814f3  run-config.json\n"
        "418dc7c21f3641baea080d3bdf66103ae8548c929a0d105305120615ecbdc512  seed-1/evaluation-per-episode.jsonl\n"
        "db7e0189b8d900a71e0410c229241befab2e0d28a81f51accb0bb32c2195a555  seed-1/selected-checkpoint.pt\n"
        "b593c53b1d7d1d6de031ec3e04ea65c1a6ce63121a01acfc21d8fe80866aaa06  seed-1/summary.json\n"
        "95c0a36f0229b030a84be2e7fb3bc9a2f09a32e55d7827904bea6e46a137e5cf  seed-1/train-steps.jsonl\n"
    )
    assert manifest_path.read_text(encoding="ascii") == expected_manifest
    assert hashlib.sha256(expected_manifest.encode("ascii")).hexdigest() == (
        "1694a3a0c23e71b4c830d432234237498666c832b579ce62f2622b007b70fe5f"
    )


_verify_gate9d_stage1_seed1_result_when_present()

from . import gate9d_fast_harness_contract as _gate9d_fast_harness_contract
from . import gate9d_query_capacity_contract as _gate9d_query_capacity_contract
