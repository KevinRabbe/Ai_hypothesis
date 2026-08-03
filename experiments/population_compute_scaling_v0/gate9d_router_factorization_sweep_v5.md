# Gate-9D router factorization sweep v5

Development-only supervised routing formulation test.

Variants:

- `decoupled_raw64`: separate bias and contribution networks;
- `factorized_overlap16`: contribution sees worker bits plus local worker/query overlap;
- `local_summary_linear`: linear gates over zero, worker popcount, and overlap.

All variants use the complete 65,536-state domain, the exhaustive margin objective, three seeds, and 512 fixed steps. This does not use answer loss, support outputs, operator identity, automatic coordinate discovery, or population confirmation.
