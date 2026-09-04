# Capability Traffic Model: completed rewrite

> **Current report.** The former `capability_*` figures are historical retained
> artifacts from removed speed-first/lane-first semantics. They are not current
> evidence. The current 30-seed results are in
> [risk_aversion_results.md](risk_aversion_results.md).

## Capability tick

The pipeline is: committed positions → bounded observations → binding LR/lane
decision → risk-adjusted speed proposal on that lane → synchronous micro-step
collision detection → successful-driver traces and replacement → acquired state
and logging.

`propose_lanes!` maps LR's sign, subject to the existing lane-error process, to
the final `LaneProposal` for the tick. Speed reads but never changes that lane;
there is no alternative-lane search. Each car attempts `max_speed` (normally 3),
makes exactly one seeded draw, and accepts danger when
`rand(rng) > risk_aversion`. If danger is rejected, progressively lower speeds
are tested on the binding lane. If all speeds 1 through `max_speed` are
dangerous, speed 1 is the unavoidable-risk fallback; stopping is not added.

Perceived danger is exact path conflict against other cars' extrapolated
committed current motion: the same cell at the same micro-step, a position
exchange, or a diagonal crossing while changing lane. Private
`LaneProposal`, `SpeedProposal`, and `MovementPath` values are invisible.
Simultaneous private proposals can therefore conflict and crash. Detection is
symmetric and has no winner.

## Risk and replacement

`RiskAversion` is mandatory in [0, 1]. Initial values and `EntryDraw`
replacement values are Uniform(0, 1), and all newborns have maximum speed.
Evolutionary replacement uniformly selects one survivor and inherits risk
aversion and capabilities from that same parent, with Gaussian trait mutation
of scale `trait_mutation_scale`, clamped to [0, 1] (zero remains exact).
Acquired states reset. Logger observables include risk means, standard
deviations and distributions; survivor and selected-parent risk; dangerous
max-speed proposal acceptance counts and rate; mean proposed and realized
speed; and replacement pressure.

See the current experiment report for design, artifacts, and interpretation.
