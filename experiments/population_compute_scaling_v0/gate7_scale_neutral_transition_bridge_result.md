# Gate-7 v0 — scale-neutral scorer transition bridge result

## Status

**FINAL FRESH LOW-SCALE TRANSITION EVIDENCE — `GATE7_SCALE_NEUTRAL_TRANSITION_QUALIFIED`.**

Measured scientific head:

`d149055978cab01346798b4442bea0bc47b46805`

The first and only admitted 256-world scale-neutral transition bridge completed with:

- `artifact_valid = true`;
- `errors = []`;
- training performed during bridge: **false**;
- checkpoint selection performed: **false**;
- high-scale Gate-7 opened: **false**;
- independent transition outcome: `GATE7_SCALE_NEUTRAL_TRANSITION_QUALIFIED`.

All twelve preregistered primary criteria passed. The scale-neutral checkpoint family is therefore qualified for a separately versioned high-scale Gate-7 routing-bandwidth campaign.

## Exact checkpoint identities

Transition checkpoints:

```text
T0 SHA256 be1b18e22a1f713b48de8934aca1c0302f1255342684158c6ae31a7c3618a719
   fingerprint 0be5c26d1056da1bf12a53be5ba5e6d1cadb7815eb117e2a77db269391c1c5aa

T1 SHA256 a9b712736f440168ccd86ee070b8d301be0eb834b9d17418d40c0175207cadfb
   fingerprint b9685382992cb2f94454b6faa3675d458f236e7b0c7d8399bea256bedcb02e46

T2 SHA256 cd7869511d07679cb1ad430743cad031299a575413e8a853ef9116d545c7480a
   fingerprint 1bca7012c7350c3b3fe8c9790a65c36eea8fcd8bef3e123034817ff78673a7bb
```

Seed-matched original controls:

```text
O0 SHA256 e63748a1182472d03c540f9123c3442ce44e130519e2176681034568826db590
   fingerprint e4f9990f08e85393a155637cfa50b5831d779770447ec3716fc9e67946992afc

O1 SHA256 8ce384627f5543fa4fe78498f9990f8214cf2a4afed0c9b734e86761ad13c989
   fingerprint 2a57984f755ee1404fd828c2df36a5efff8f13a541b7d7c93a891014fbf4897c

O2 SHA256 103826bbd9451b965eced5134d1674cb8e893f5d3b378cb828312ffdb6fc9a37
   fingerprint 8afaf956f200f41ea914eafdd1b5f151dd303cae1552165c74e12bb8c945af02
```

Every neural condition used exactly 19,649 learned parameters. No checkpoint was discarded, ranked, retrained, fine-tuned or selected after exposure.

## Frozen bridge mechanics preserved

Per condition:

- fresh bridge-only hidden/hint/runtime/bootstrap namespaces;
- depth 10;
- hint reliability 0.70;
- populations N128 and N256;
- transition global, learned K16 and matched answer-blind hash K16;
- seed-matched original global control at N256;
- 256 paired worlds;
- evaluation batch 64;
- 2,000 deterministic paired-bootstrap samples;
- 255 common Stage-A parent slots;
- 128 Stage-B routing parent slots;
- 6,128 learned recurrent updates/world;
- compiler, CUDA graphs and mixed precision off.

The experiment contained exactly 21 condition cells and 15 paired comparisons. N256 transition K16-versus-global was descriptive only and could neither qualify nor reject the transition.

## Coverage rates

| Checkpoint | N128 global | N128 K16 | N128 hash K16 | N256 global | N256 K16 | N256 hash K16 | N256 original global |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| T0/O0 | 0.3515625 | 0.3359375 | 0.11328125 | 0.5703125 | 0.5390625 | 0.0546875 | 0.56640625 |
| T1/O1 | 0.34765625 | 0.33203125 | 0.11328125 | 0.5703125 | 0.56640625 | 0.0546875 | 0.56640625 |
| T2/O2 | 0.34765625 | 0.34375 | 0.11328125 | 0.58203125 | 0.546875 | 0.0546875 | 0.56640625 |

## Learned K16 versus matched hash control

### N128

```text
T0 +0.22265625 CI [0.1640625, 0.28125]
T1 +0.21875    CI [0.16015625, 0.2734375]
T2 +0.23046875 CI [0.171875,   0.2890625]
```

### N256

```text
T0 +0.484375   CI [0.41015625, 0.55859375]
T1 +0.51171875 CI [0.44140625, 0.58203125]
T2 +0.4921875  CI [0.421875,   0.55859375]
```

Every learned-routing CI low is strictly above zero. The transition checkpoints preserve the established learned routing mechanism strongly and consistently at both bridge populations.

## K16 non-inferiority to transition global at N128

Frozen non-inferiority margin: `0.05`.

```text
T0 -0.015625   CI [-0.03515625, 0.00390625] PASS
T1 -0.015625   CI [-0.03515625, 0.0]        PASS
T2 -0.00390625 CI [-0.0234375,  0.01171875] PASS
```

All three CI lows remain above `-0.05`.

## Transition global versus original global at N256

```text
T0 +0.00390625 CI [-0.01171875, 0.0234375] PASS
T1 +0.00390625 CI [-0.01171875, 0.0234375] PASS
T2 +0.015625   CI [-0.0078125,  0.0390625] PASS
```

The scale-neutral global scorer is non-inferior to the seed-matched original global scorer on every checkpoint. Point estimates are slightly positive in all three cases.

## Descriptive N256 K16 versus transition global

```text
T0 -0.03125
T1 -0.00390625
T2 -0.03515625
```

This comparison was intentionally descriptive because Gate-6 already showed checkpoint-sensitive near-global K16 behavior at N256. It does not affect transition qualification. Descriptively, the new checkpoint family nevertheless kept K16 within 3.6 percentage points of global on every seed.

## Scientific interpretation

The bridge establishes that replacing the depth-limited one-hot representation with the frozen scale-neutral representation did not destroy the previously established routing mechanism.

Within the controlled depth-10 bridge regime:

- learned K16 routing remains substantially better than an answer-blind matched K16 control;
- K16 remains near-global at N128 under the inherited five-point margin;
- the scale-neutral global reference remains viable and seed-robust at N256;
- the three independently trained transition checkpoints behave consistently enough that no checkpoint-specific rescue or selection is needed.

This qualifies the representation/checkpoint transition only. It does **not** yet establish how routing visibility must scale for N greater than 256, an asymptotic law for `K_required(N)`, or capability-per-compute superiority.

## Provenance

- result SHA-256: `705a9d85ce6059f36de7d7bfaf47db0fab342f85662fd06be61514383ca08884`
- independent audit SHA-256: `c6772d454964f92bd30f806b04af9056fbfe34d691807678c3a4b8dde6d80590`
- recursive manifest SHA-256: `518af71206972925b3ed13c4791011f19a9a49899f737f0ff750ae0294069bdb`
- output root: `F:\gate7_scale_neutral_transition_bridge_v0`
- transition-training invocation head: `07307650b2bbbfaa09b80e40caa4419ecdda2947`
- measured bridge head: `d149055978cab01346798b4442bea0bc47b46805`

## Stop and continuation rule

The scale-neutral transition bridge v0 is closed with `GATE7_SCALE_NEUTRAL_TRANSITION_QUALIFIED`.

Do not:

- rerun an alternate bridge namespace;
- change the 0.05 margin;
- select or discard checkpoints;
- retrain or fine-tune the transition checkpoints;
- reinterpret descriptive N256 K16-versus-global as a primary criterion;
- add post-result bridge conditions.

A high-scale routing-bandwidth study may now be admitted only as a separately versioned Gate-7 scientific protocol using these exact three transition checkpoints and a fresh high-scale world namespace.
