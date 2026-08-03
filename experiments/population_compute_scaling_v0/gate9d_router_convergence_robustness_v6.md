# Gate9D router convergence robustness v6

Development-only supervised routing study.

The v5 factorization sweep showed the 8-parameter local-summary router and the 338-parameter overlap router are representable but seed-sensitive. This slice keeps the 8-parameter representation fixed and compares random-init AdamW, deterministic full-batch LBFGS, and an exact analytic separator.

A pass establishes reliable convergence for at least one method. It does not establish end-to-end answer-loss learning, automatic coordinate discovery, or population confirmation.
