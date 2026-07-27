from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ai_hypothesis.large_scope.validate_result import main
from tests.test_large_scope_result_contract import valid_payload


class LargeScopeValidateResultCliTests(unittest.TestCase):
    def test_valid_artifact_writes_normalized_readout_without_scientific_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            source = root / "input.json"
            output = root / "validated.json"
            source.write_text(json.dumps(valid_payload()), encoding="utf-8")

            exit_code = main(
                [
                    "--input",
                    str(source),
                    "--output",
                    str(output),
                ]
            )

            self.assertEqual(exit_code, 0)
            normalized = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(normalized["status"], "VALID")
            self.assertEqual(normalized["scientific_decision"], "NOT_ASSIGNED")
            self.assertEqual(normalized["readout"]["widths"], [1, 4, 16])
            self.assertEqual(len(normalized["readout"]["checkpoint_ids"]), 16)

    def test_wrong_expected_widths_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            source = Path(tempdir) / "input.json"
            source.write_text(json.dumps(valid_payload()), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "expected benchmark widths"):
                main(
                    [
                        "--input",
                        str(source),
                        "--expected-widths",
                        "1",
                        "4",
                    ]
                )


if __name__ == "__main__":
    unittest.main()
