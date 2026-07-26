# Permanent exploration under integration backpressure v0

## Problem

Scheduler v0 originally treated integration backpressure as a total redirect:

```text
backpressure active
    ↓
all scheduler decisions become VERIFY or SYNTHESIZE
```

That behavior was simple, but it conflicts with a locked architecture rule:

> Exploit what currently works without completely stopping the search for something better.

At large population scale, integration pressure may be persistent rather than transient. If backpressure disables discovery completely, a sufficiently large evidence backlog can permanently convert the whole population into integration workers.

That creates premature convergence at the system level even if enormous neural compute remains available.

## v0 rule

Backpressure changes the compute mix; it does not reduce exploration probability to zero.

Scheduler v0 now has two lanes while integration backpressure is active:

1. **integration / verification lane** — dominant;
2. **structured exploration lane** — permanently nonzero.

Default:

```text
95%  service integration / verification pressure
 5%  structured exploration
```

The numeric 5% share is provisional. The invariant is the important part.

## Configuration

`SchedulerConfig` adds:

```text
backpressure_exploration_probability = 0.05
```

The field is appended after the existing config fields so existing positional construction keeps its original meaning.

The value must satisfy:

```text
0 < backpressure_exploration_probability < 1
```

This guarantees both lanes remain possible:

- zero is rejected because discovery would disappear;
- one is rejected because integration backpressure would never be serviced.

## Backpressure service lane

When the random draw does not enter the reserved exploration lane, Scheduler v0 preserves the existing behavior:

1. select the active thread with the greatest integration backlog pressure;
2. use normal priority as a tie-breaker;
3. if verification need is above threshold, emit `VERIFY`;
4. otherwise emit `SYNTHESIZE`.

Reason codes remain:

- `BACKPRESSURE`
- optionally `VERIFY`

The purpose-aware context router can then provide bounded pending evidence or unresolved knowledge through the ordinary Work Item contract.

## Reserved exploration lane

When the backpressure exploration draw succeeds, the scheduler uses the same structured exploration selector already used outside backpressure.

Candidate weighting still favors:

- missing coverage;
- novelty;
- uncertainty.

The resulting decision is:

```text
action  = ADD_WIDTH
purpose = EXPLORE
```

with reason codes:

- `STRUCTURED_EXPLORATION`
- `BACKPRESSURE_EXPLORATION`

Width remains bounded by the ordinary exploration width and available worker capacity.

No separate discovery scheduler is introduced.

## Why probabilistic rather than a fixed quota

v0 intentionally avoids another durable counter/state machine.

A fixed policy such as "every 20th allocation is exploration" would require persistent quota state if exact behavior had to survive restart.

The current structured-random lane keeps the scheduler stateless behind its existing contract. Long-run allocation approaches the configured share without adding another persistence primitive.

If future evidence shows that exact quota guarantees are necessary, they can be implemented behind the same SchedulerDecision boundary.

## Interaction with knowledge-integration bandwidth

PR #33 adds direct telemetry for:

- evidence generation rate;
- unique evidence absorption rate;
- backlog growth;
- backlog age;
- redundant integration traffic.

Those measurements can later answer whether the default exploration share is too high or too low.

Examples:

### Integration comfortably keeps up

The exploration share may be increased if additional possibility coverage is valuable.

### Backlog grows continuously

The share may be reduced, integration width may be increased, or integration itself may need to scale hierarchically.

### Integration is saturated despite exploration near zero

Reducing discovery further is not a solution; the integration architecture itself is the bottleneck.

v0 does not tune the probability from those signals automatically.

## Architecture invariant

The important rule is not "5% exploration."

It is:

> Persistent information pressure may reduce exploration, but must not eliminate it.

This protects the population from locking itself into its current knowledge simply because integration is expensive.

## Non-goals

This slice does not add:

- a learned scheduler;
- adaptive probability tuning;
- a token-bucket/quota scheduler;
- a second exploration scheduler;
- hierarchical integration;
- worker specialization;
- new persistence state;
- a claim that 5% is optimal.

The numeric policy remains replaceable. The permanent dual-lane behavior is the architectural direction.
