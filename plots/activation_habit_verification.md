# Activation timing and habit: verification report

This report evaluates 30 paired seeds (20260801–20260830). Each seed was run under all four timing × habit conditions for 600 ticks with 120 cars on a 2×300 torus. Metrics exclude the first 150 ticks. Lookahead=60, error rate=0.01, and active habit weight=0.5.

## Controlled mechanism check

For two opposing cars approaching the same cell, the collision matrix is:

| focal \ other | right | left |
|---|---:|---:|
| right | safe | collision |
| left | collision | safe |

Thus the encounter is an anti-coordination problem: both actions can be safe, but neither is safe independently of the other driver's simultaneous choice. This verifies indeterminacy, not logical impossibility.

A one-tick mechanism fixture then gives:

| Timing | Habit state | Failed encounter |
|---|---|---:|
| Sequential | none | 0 |
| Simultaneous | none | 1 |
| Simultaneous | exogenously aligned | 0 |

Sequential activation lets the second driver react to the first driver's committed move. With simultaneous activation that ordering signal disappears. A sufficiently strong, shared habit restores coordination in the fixture. Because the habit is deliberately aligned here, this is a mechanism test—not evidence that alignment emerges by itself.

## Population hypotheses

Condition means across paired replications:

| Timing | Habit | Compatibility | Pre-coordination | Disposition alignment | Disposition predictability | Failure rate |
|---|---|---:|---:|---:|---:|---:|
| sequential | false | 0.9802 | 0.97043 | 0.99104 | 0.98425 | 0.0198 |
| sequential | true | 0.98051 | 0.96492 | 0.98524 | 0.98189 | 0.01949 |
| simultaneous | false | 0.97831 | 0.97126 | 0.99249 | 0.98546 | 0.02169 |
| simultaneous | true | 0.98022 | 0.97186 | 0.99125 | 0.98591 | 0.01978 |

| ID | Falsifiable claim | Paired effect | 95% bootstrap CI | one-sided p | Result |
|---|---|---:|---:|---:|---|
| H1 | simultaneous timing reduces compatible joint choices without habit | -0.00189 | [-0.00267, -0.00115] | 5.0e-5 | supported |
| H2 | habit increases compatible joint choices under simultaneous timing | 0.00191 | [0.00106, 0.00276] | 0.0001 | supported |
| H3 | habit increases pre-coordinated encounters | 0.0006 | [-0.00491, 0.0047] | 0.43298 | not supported |
| H4 | habit improves compatibility more under simultaneous than sequential timing | 0.00161 | [0.00074, 0.00253] | 0.0007 | supported |

Effects are first condition minus second condition as encoded by each claim; H5 is the difference-in-differences.

## Interpretation guardrails

Habit increased compatible joint choices conditional on cars meeting. This establishes an encounter-level coordination gain, but does not by itself show that the choice was settled in advance.
Pre-coordinated encounters did not increase reliably. The compatibility gain therefore does not, on its own, verify the proposed reduction of strategic uncertainty through prior habit.

The p-values are paired sign-randomization estimates and the intervals are paired bootstrap intervals. They are exploratory (no multiple-testing correction) and should be accompanied by sensitivity checks over density, error rate, lookahead, habit strength, and replacement policy.
