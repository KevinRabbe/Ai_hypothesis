# Gate 2 — execution semantics supplement v0

Status: **FROZEN BEFORE ANY GATE-2 TRAINING OR DEVELOPMENT RESULT**

This file removes implementation ambiguity from `gate2_persistent_state_capacity_protocol_v0.md`. It does not change the scientific question, frozen matrix, or positive-evidence rule.

## Width-independent observation stream

For one world, the encoded observation tensor is constructed before runtime-width organization is applied.

Every width/control receives exactly the same ordered round-major stream:

- 4 evidence rounds;
- 4 interference/retention rounds;
- every entity exactly once per round;
- no target-query identity inside the observation stream;
- no payload information in interference fields.

The final query key and answer payload remain outside the observation stream.

## Persistent state bank

A condition with width `W` owns exactly `W` temporary neural state vectors.

The learned update rule is one shared `GRUCell` reused for every state vector. There are no slot-specific learned parameters and no learned router.

For every world with `C` entities, each of the 8 rounds performs exactly `C` learned recurrent updates. Therefore every width/control performs exactly `8 × C` learned recurrent updates.

## Collision-lane ordering

Routing is balanced. For each round and slot, entities assigned to that slot are ordered by canonical entity index.

If collision load is `L = C / W`, execution proceeds through `L` collision lanes:

1. lane 0 updates each slot from its first routed entity;
2. lane 1 updates each slot from its second routed entity;
3. continue through lane `L-1`.

The parallel schedule updates all `W` independent slots together within one lane. The serial schedule time-multiplexes those same slot-local update sequences one slot at a time.

Because slots do not communicate during this workload, parallel and serial schedules must produce the same mathematical state bank and decoded output. A capability difference between schedules is a correctness failure.

## Reset-state control

`reset_state` uses the exact stable routing and exact observations of `stable_persistent`, but the complete state bank is reset to zero at each round boundary after round 0.

It therefore preserves width, information, routing, and learned update count while removing cross-round memory.

## Reshuffled-locality control

`reshuffled_locality` uses an independently domain-separated balanced entity permutation on every round.

It preserves:

- width;
- per-round slot load;
- observation stream;
- learned update count;
- state-bank size.

It changes only which persistent slot receives a given entity on later rounds.

At width 1 the routing tensor is exactly identical to stable routing and must produce bit-for-bit identical execution.

## Final delayed query

The readout receives:

- the queried entity key;
- the state vector in the slot that received that entity in the final retention round.

For stable/reset routing this is the stable target slot. For reshuffled locality it is the target's final-round slot.

This choice gives the reshuffled control access to the target's current local state without secretly restoring the earlier stable trajectory. No additional query-time routing permutation is introduced after the eight frozen update rounds.

## Model/readout boundary

Development starts with:

- state width `64`;
- query projection width `24`;
- 4 output logits, one per payload bit;
- exact payload decoding from the sign of the four logits.

These values are development architecture choices and are not confirmation-frozen until the development stage is completed. They must be frozen, together with optimizer/training recipe and numerical-equivalence policy, before untouched confirmation worlds or new confirmation training seeds are opened.

## Current evidence boundary

The mechanics and schedule-equivalence tests may execute on synthetic fixtures/random untrained weights. They are not a Gate-2 capability result.

No development training curve or held-out capability result exists at the time this supplement is frozen.
