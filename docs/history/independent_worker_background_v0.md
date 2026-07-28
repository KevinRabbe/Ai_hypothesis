# Independently Weighted Worker Program — Historical Background

## Status

**Superseded as the primary research direction. Preserved as background and failure-analysis history.**

The current project does not attempt to build a population by adding independently trained miniature models whose learned parameter capacity grows with worker count.

The canonical direction is one fixed/shared learned parameter set reused across many temporary runtime neural states.

## What the earlier work established

Earlier experiments explored how small an independently trained local neural worker could become while still performing useful local transformations.

The repository's surviving canonical summaries place the practical observed region roughly at:

- around 10K parameters: difficult;
- around 25K: smallest strong candidate region;
- around 50K: practical reference used in later experiments;
- around 75K–100K: larger reference points.

These observations remain useful background: nontrivial learned local transformations do not necessarily require a large standalone model.

They do **not** establish that a population of those independent models is a parameter-efficient organism.

## Why the direction became scientifically insufficient

With independently trained checkpoints:

```text
1 worker  → P learned parameters
4 workers → approximately 4P learned parameters
16 workers → approximately 16P learned parameters
```

Even when all workers share an architecture/training distribution, increasing population also increases total learned capacity.

That makes a positive population-width curve ambiguous:

- did runtime organization help?;
- did source coverage help?;
- did independent weight diversity help?;
- or did the system simply receive more learned parameters?

The fixed-parameter population-compute program was introduced to remove that confound.

## Step-2 minority-evidence question

One later independently weighted-worker direction focused on the gap between:

- a population containing at least one locally correct worker; and
- a reducer actually selecting that minority-correct answer.

The key architectural concern was that averaging/voting style population reduction could erase rare useful evidence.

A minority-rescue diagnostic therefore explored whether inference-visible evidence could distinguish a genuinely useful minority candidate from a noisy outlier without changing the existing reducer prematurely.

This remains useful failure-analysis history because it identified a general principle:

> Population computation gains are useless if aggregation systematically destroys rare decisive information.

The current shared-weight organism should retain that lesson without inheriting the old reducer/model stack.

## Evidence versus conclusion

The historical work increasingly moved from discrete worker votes toward structured/continuous evidence.

Useful retained principle:

- a worker/local state should emit evidence or bounded transformations;
- population integration should not force every local contribution immediately into one hard class vote;
- minority/contradictory evidence should remain inspectable until a causal integration rule resolves it.

This later influenced the persistent-runtime evidence/knowledge architecture, although that runtime is itself now deferred.

## Why shared weights are cleaner

The current research organism uses one learned update rule/checkpoint while changing runtime state/population size.

Conceptually:

```text
fixed learned weights
      ↓
shared neural machinery
      ↓
state 1
state 2
state 3
...
state N
```

Increasing `N` changes:

- temporary state memory;
- neural update count;
- communication;
- available information scope;
- parallel execution organization.

It does **not** automatically increase learned parameter count.

That makes the central causal question much cleaner:

> Can a fixed learned system convert more runtime computation/state into more capability?

## What remains reusable from the old worker program

### Small local learned transformations

The earlier size sweeps support continuing to investigate weak local neural processing elements rather than assuming every runtime unit must be a complete model.

### Independent diversity as a future controlled variable

Different learned parameter sets may still be scientifically interesting later, but only under a fixed total learned-parameter budget or another explicit matched-capacity control.

Do not reintroduce one independent full checkpoint per runtime worker and call the result fixed-parameter population scaling.

### Minority information preservation

Any future communication/reduction mechanism should be tested for whether rare decisive local information survives population integration.

### Same-scope controls

The later large-scope program correctly separated source coverage from weight diversity. That experimental pattern remains reusable.

## What should not be carried forward automatically

Do not restore as canonical defaults:

- independently trained full worker checkpoints per runtime state;
- reducer-v0 or minority-rescue thresholds;
- population-wide voting as the main collective mechanism;
- a target worker size such as 50K merely because it was previously practical;
- worker autonomy as a requirement.

The current unit may be much weaker than the earlier worker. Its usefulness may exist only as part of the population dynamics.

## Historical interpretation boundary

The old results answer a different question from the current Gate program.

They provide evidence about:

- feasible size of local learned transformations;
- failure modes of population evidence reduction;
- practical handling of independently weighted worker banks.

They do not establish:

- fixed-parameter capability scaling;
- capability advantage of simultaneous population state;
- parameter efficiency versus one shared/recurrent model;
- general intelligence scaling.

Those questions are now handled by the shared-weight Gate-0/1/2 sequence.

## Reactivation condition

Independent learned diversity should return only if a later experiment asks an explicit fixed-budget question such as:

> Given the same total learned-parameter budget, is it better to allocate all parameters to one shared organism or partition some capacity across a small number of independently learned specialists/lineages?

That would be a new controlled experiment, not restoration of the historical default.
