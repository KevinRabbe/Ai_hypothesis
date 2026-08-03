# Gate-9D router architecture sweep v4

**Development only.** This sweep keeps exhaustive supervised routing labels, the exact finite-domain margin objective, and the fixed XOR population executor. It does not use answer loss, support outputs, operator identity, automatic coordinate discovery, or population confirmation.

Variants:

- `raw_width128`: wider one-hidden-layer raw-bit router;
- `raw_deep64`: two-hidden-layer raw-bit router;
- `interaction16`: compact router with local elementwise worker/query interaction features.

All three variants train on all 65,536 local routing states for 512 steps across the same three initialization seeds. The first variant with strict bias and contribution separation in all three seeds is the development winner.
