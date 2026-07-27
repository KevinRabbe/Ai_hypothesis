from __future__ import annotations

import unittest

from ai_hypothesis.large_scope import (
    LargeScopeResultReadout,
    LargeScopeWidthReadout,
    validate_large_scope_result,
)


class LargeScopePublicResultContractTests(unittest.TestCase):
    def test_result_contract_is_available_from_public_large_scope_api(self) -> None:
        self.assertTrue(callable(validate_large_scope_result))
        self.assertEqual(LargeScopeResultReadout.__name__, "LargeScopeResultReadout")
        self.assertEqual(LargeScopeWidthReadout.__name__, "LargeScopeWidthReadout")


if __name__ == "__main__":
    unittest.main()
