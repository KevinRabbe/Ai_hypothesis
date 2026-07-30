# Gate-6 v0 — fixed-K population scaling development result

## Status

**VALID DEVELOPMENT EVIDENCE — `G6_S3_CHECKPOINT_SENSITIVE_SCALING`.**

Measured scientific head:

`02671dcd070201690aa71f7326b1f0779bc660c4`

The first and only admitted 256-world Gate-6 development matrix completed with:

- `artifact_valid = true`;
- `errors = []`;
- training performed: **none**;
- confirmation opened: **false**;
- independent directional outcome: `G6_S3_CHECKPOINT_SENSITIVE_SCALING`.

Per the frozen Gate-6 protocol, only `G6_S2_ROBUST_FIXED_K_POPULATION_SCALING` could open a separately versioned confirmation. Therefore **Gate-6 v0 confirmation remains closed** and no post-result K/checkpoint/margin rescue is permitted.

## Frozen mechanics preserved

All conditions reused the exact three frozen 19,649-parameter checkpoints with no training/fine-tuning.

Per world and condition:

- task depth: 10;
- hint reliability: 0.70;
- population ladder: N64 / N128 / N256;
- common Stage A: 255 parent expansions = 4,080 learned recurrent updates;
- Stage B: 128 parent expansions = 2,048 learned recurrent updates;
- total learned recurrent work: **6,128 updates/world**;
- two active child lanes;
- eight recurrent updates/child;
- K16 primary bounded-score treatment;
- K16 answer-blind hash control;
- K8 descriptive only;
- 2,000 deterministic paired bootstrap samples.

## Primary pass matrix

`PASS(C,N)` is frozen as both:

1. `K16 - hashK16` paired-bootstrap CI low > 0; and
2. `K16 - global` paired-bootstrap CI low > -0.05.

| Tier | C0 | C1 | C2 |
| --- | --- | --- | --- |
| N64 | PASS | PASS | PASS |
| N128 | PASS | PASS | PASS |
| N256 | PASS | **FAIL NI** | **FAIL NI** |

Thus PASS status is mixed across checkpoints at N256, mechanically selecting the frozen higher-precedence outcome `G6_S3_CHECKPOINT_SENSITIVE_SCALING`.

## Learned K16 routing versus matched hash control

### N64

```text
C0 +0.0859375  CI [0.05078125, 0.125]
C1 +0.08984375 CI [0.0546875,  0.125]
C2 +0.078125   CI [0.04296875, 0.11328125]
```

### N128

```text
C0 +0.296875   CI [0.234375,   0.36328125]
C1 +0.29296875 CI [0.23046875, 0.35546875]
C2 +0.29296875 CI [0.234375,   0.3515625]
```

### N256

```text
C0 +0.48046875 CI [0.4140625, 0.55078125]
C1 +0.484375   CI [0.4140625, 0.55078125]
C2 +0.4453125  CI [0.37890625, 0.51171875]
```

The learned-routing signal therefore remains strongly positive at every tested checkpoint/tier. Gate-6 did **not** observe K16 becoming harmful or useless relative to the matched answer-blind bounded control.

## K16 non-inferiority to global routing

Frozen NI margin: `delta_NI = 0.05`.

### N64

```text
C0 -0.01953125 CI [-0.0390625,  -0.00390625]  PASS
C1 -0.01953125 CI [-0.0390625,  -0.00390625]  PASS
C2 -0.0234375  CI [-0.04296875, -0.0078125]   PASS
```

### N128

```text
C0 -0.01171875 CI [-0.03125,    0.0078125]    PASS
C1 -0.01953125 CI [-0.0390625, -0.00390625]  PASS
C2 -0.0234375  CI [-0.04296875,-0.0078125]   PASS
```

### N256

```text
C0 -0.0078125  CI [-0.03515625,  0.01953125]  PASS
C1 -0.03125    CI [-0.06640625,  0.00390625]  FAIL
C2 -0.05078125 CI [-0.08984375, -0.015625]    FAIL
```

At N256 the primary K16 treatment remains highly useful, but the stronger five-percentage-point near-global claim no longer holds robustly across checkpoints.

## Score-observation scaling

K16 remained exactly bounded at:

`128 Stage-B slots × 16 visible scores = 2,048 score observations/world`

for every N and checkpoint.

Global score observations rose strongly with population:

```text
N64:  ~5,978–5,993 / world
N128: ~14,516–14,579 / world
N256: ~30,871–31,072 / world
```

Thus the information-saving property remained real even where the strict NI criterion became checkpoint-sensitive.

## Descriptive K8 frontier

K8 used exactly 1,024 Stage-B score observations/world. It was already marginal/below the frozen 5pp NI line at some smaller-tier checkpoints and degraded substantially at N256. Because K8 was descriptive only, it does not affect the Gate-6 outcome.

## Scientific interpretation

Gate-6 identifies a controlled bandwidth/population frontier rather than a collapse of learned routing.

The evidence supports:

- fixed K16 learned routing remains useful relative to an answer-blind K16 control from N64 through N256;
- K16 retains the frozen five-point near-global criterion robustly through N128;
- at N256, near-global status becomes checkpoint-sensitive under the fixed K16 visibility budget.

This is evidence **against** the stronger hypothesis that constant K16 remains universally sufficient as population grows over the tested range.

It motivates a separately versioned next experiment asking how required score visibility K scales with N (for example whether K32/K64 can restore near-global routing at larger populations), rather than retuning Gate-6 after exposure.

## Provenance

- result SHA-256: `c1fef3338c19704a0153e1b6dc789aff1f03eed7648953f33ed72a0020a7d961`
- independent audit SHA-256: `37e886d18c48a57219135db50f859342d8d0168c40ff6d10e7607ddc712e24ca`
- recursive manifest SHA-256: `f2d5dba76fd26eaadf7011fa341cf27a89b59a7a26eab2d6942f8708fe519344`
- output root: `F:\gate6_fixed_k_population_scaling_development_v0`

## Stop rule

Gate-6 v0 is closed at development stage with `G6_S3_CHECKPOINT_SENSITIVE_SCALING`.

Do not:

- change K16;
- add K32/K64 conditions inside Gate-6;
- alter the 0.05 NI margin;
- select checkpoints;
- rerun alternate development namespaces;
- open Gate-6 confirmation.

Any follow-on bandwidth-scaling study is a new gate/protocol.
