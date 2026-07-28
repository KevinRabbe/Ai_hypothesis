# Gate 2 development execution note

Status: **DEVELOPMENT RUNNER QUALIFIED; NO DEVELOPMENT RESULT YET; CONFIRMATION LOCKED**

Qualified runnable head:

`bb08bff9fcfe9b2bcdddfdb4b1a56789869c16d6`

The development layer now provides:

- one shared checkpoint trained only on the stable-persistent mechanism;
- a deterministic cycle across all 12 frozen stable `entity_count × width` conditions;
- training-world seeds isolated below `2^30`;
- held-out development worlds isolated in `[2^30, 2^31)`;
- confirmation worlds isolated in `[2^31, 3 × 2^30)` and rejected unless explicitly unlocked by non-development code;
- one unchanged checkpoint evaluated over all 36 `entity_count × width × control` cells;
- exact solve, bit accuracy, collision load, information/work accounting and checkpoint identity per cell;
- paired stable-width-vs-width1, stable-vs-reshuffled and stable-vs-reset summaries on identical worlds;
- deterministic paired-bootstrap confidence intervals for development diagnostics;
- no development p-value/effect threshold that assigns a Gate-2 verdict.

The local runner is:

`scripts/run_gate2_development.ps1`

Default first probe:

- training seed `0`;
- 1,000 optimizer steps;
- training batch size `32`;
- 256 held-out development worlds per entity-count tier;
- evaluation batch size `64`;
- 2,000 paired bootstrap samples;
- eager CUDA;
- state width `64`, query width `24`.

The runner requires a clean Git tree, verifies CUDA availability, records Git/GPU/runtime/config provenance, saves the development checkpoint and JSON result, and writes a SHA-256 manifest. It exposes no confirmation-opening option.

This first run is intentionally a learnability/development probe. Its result may justify changing the training recipe. Any such change must remain development-only. Architecture, optimizer, final evaluation recipe, numerical equivalence rule and confirmation decision rule must be frozen before untouched confirmation worlds or new confirmation training seeds are opened.
