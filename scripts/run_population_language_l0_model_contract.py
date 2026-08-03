from __future__ import annotations

import json
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ai_hypothesis.population_language import l0_protocol as protocol
from ai_hypothesis.population_language.l0_data import materialize_batch
from ai_hypothesis.population_language.l0_models import validate_model_parameter_counts


def main() -> int:
    protocol_report = protocol.validate_protocol()
    model_report = validate_model_parameter_counts()
    batch = materialize_batch("train", (0, 1, 2, 3))
    report = {
        "status": "POPULATION_LANGUAGE_L0_MODEL_CONTRACT_ONLY",
        "protocol": protocol_report,
        "models": model_report,
        "batch": {
            "input_shape": list(batch.input_ids.shape),
            "target_shape": list(batch.target_ids.shape),
            "loss_tokens": int(batch.loss_mask.sum().item()),
            "answer_tokens": int(batch.answer_mask.sum().item()),
        },
        "training_result_claimed": False,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if protocol_report["valid"] and model_report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
