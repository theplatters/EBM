# Redesigning the Traffic-Convention Extension

## Purpose

The current Traffic model contains a useful replication core, but its forecast-strategy extension has moved away from the substantive question posed by Hodgson and Knudsen. The strongest strategies improve survival by supplying agents with increasingly accurate forecasts of other agents' next actions. That turns the model into a search for a better collision-avoidance algorithm. The reference model instead asks how boundedly rational agents, heterogeneous dispositions, selection, and acquired habit can generate and stabilize a convention when calculation alone is insufficient.

The software design compounds this conceptual shift. Forecast behavior is represented by a mutually exclusive strategy assigned to each car, while every car otherwise retains essentially the same component signature. The implementation therefore uses ECS storage and queries, but does not yet make behavioral composition the scientific object of the model.

This memo diagnoses those problems, retracts an earlier proposal to add a generic `Deliberating` capability, and proposes a smaller extension built around observable speed, local convention perception, optional model-specific capabilities, and staged synchronous interaction.

The central recommendation is:

> Preserve the original lane-evaluation mechanism as bounded calculation. Do not try to solve simultaneous lane choice by forecasting private intentions. Add an observable speed/yield decision and a lagged, locally perceived convention signal. Represent the mechanisms that produce each decision as optional ECS components.

| Requirement | Design response |
|---|---|
| Utilize ECS fully | Replace the strategy enum with optional, model-specific capability and acquired-state components processed by separate systems. |
| Parallel agent interactions | Compute private proposals from one committed snapshot, resolve conflicts deterministically, and commit simultaneously. |
| Bounded rationality and information | Restrict decisions to local physical observations, personal state, and lagged perceived convention; never expose other agents' dispositions or current proposals. |
| Break symmetry | Add observable integer speeds from 1 to 3, retain path-dependent personal habit, and use locally observed past convention only as a tie-breaker. |

## 1. What the reference model is about

Hodgson and Knudsen describe heterogeneous, boundedly rational drivers. Drivers observe traffic in a limited region ahead and combine same-direction traffic, opposite-direction traffic, near-field avoidance, and acquired habit. Drivers do not know other drivers' sensitivities, habits, or intended moves. Collision and replacement generate selection over fixed behavioral dispositions, while repeated action changes habituation during a driver's lifetime.

The significant result is not that habit is the most accurate predictor. It is that habit can improve or sustain a convention alongside calculation. A convention reduces the practical need to solve every encounter independently. Repeated individual behavior generates an aggregate regularity, and that regularity feeds back into the experiences and dispositions of individuals. This is the model's connection to reconstitutive downward causation.

Accordingly, collision reduction is primarily systemic:

```text
bounded local observation
        ↓
initially inconsistent choices
        ↓
small contingent population imbalance
        ↓
repeated action and differential survival
        ↓
individual habituation
        ↓
shared left/right convention
        ↓
fewer opposing cars enter the same physical lane
```

No driver needs to predict every other driver's next action for this process to work.

## 2. Problems with the forecast strategies

### 2.1 The optimization target has displaced the theoretical target

The strategy analysis ranks policies largely by survival, replacement counts, and lane switching. These are relevant outcomes, but optimizing them favors mechanisms that give agents more model knowledge. The resulting leaderboard answers “which centralized forecast produces the fewest simulated crashes?” more directly than “when and why does habit support convention formation under bounded rationality?”

An oracle may be useful as a performance bound. It should not become the principal behavioral result merely because it performs best.

### 2.2 Decision-aware prediction assumes away the problem

`DecisionAwareStrategy` gathers the positions, directions, sensitivities, avoidance parameters, habit genes, and habituation values of the population. It repeatedly applies the model's decision equation until its probabilistic lane forecast approaches a damped best response.

This conflicts with the reference model in three ways:

1. **Private state becomes public information.** A driver effectively knows dispositions and acquired habits that it cannot observe.
2. **The model of other agents is known.** Drivers behave as though they know that others use the same decision equation and know its parameters.
3. **Mutual prediction is resolved centrally.** Iteration supplies a computational fixed point where bounded agents face an unresolved coordination problem.

The strategy's good performance is therefore unsurprising but substantively uninformative. It is closer to an oracle or rational-expectations benchmark than a plausible driver capability.

### 2.3 The other shared forecasts also have epistemic problems

- **Per-entity habitus** forecasts other cars from their private acquired habits.
- **Mean habitus** exposes an exact population-wide statistic of private dispositions.
- **Random and Switch** are useful stress tests but not serious behavioral mechanisms.
- **Naive** uses only observable current lanes, but treats constant-lane continuation as a common point forecast.
- **Two-frame Naive** is more restrained than Decision-aware and uses current information, but its temporal-consistency rule remains an ad hoc survival improvement rather than an extension of the theory of habit.

The single shared `PredictedOccupancy` resource also gives all cars a common belief. Physical occupancy can legitimately be shared environmental state. A forecast is an agent-specific epistemic state and should not silently become objective world state.

### 2.4 Simultaneous lane prediction has no general local solution

When two otherwise symmetric drivers choose lanes simultaneously, each driver's safest action depends on the other's action. More recursive calculation does not generate missing information. A deterministic local predictor can oscillate, while independent randomization merely assigns a probability to successful coordination.

The shared left/right convention resolves the encounter by correlating behavior, not by making each driver a superior forecaster. This is precisely why the model should not be reframed around increasingly sophisticated intention prediction.

## 3. Why ECS is currently underused

The heterogeneous extension stores a `DriverStrategy` value on every car. A strategy kind selects a monolithic forecasting policy. This reproduces an agent-type design inside an ECS component:

```text
car → one strategy label → one behavior branch
```

The result has several limitations:

- Strategy combinations are not possible: a car cannot independently possess habit learning, convention perception, and speed adjustment.
- The enum becomes a hidden class hierarchy.
- Systems branch on strategy identity instead of selecting entities by required state.
- Stable dispositions, learned state, observable physical state, and temporary decisions are not clearly separated.
- Replacement preserves strategy composition, preventing the selection process that is central to the reference model.
- Dynamic component changes risk becoming demonstrations of ECS syntax rather than theoretically justified behavioral transitions.

Using ECS fully does not mean maximizing the number of components or adding and removing markers every step. It means that each component has an explicit scientific interpretation and each system implements one process for all entities possessing the required state.

## 4. Why the first capability proposal was wrong

The earlier proposal introduced generic capabilities such as perception, memory, avoidance, imitation, and deliberation. It was not sufficiently disciplined.

### 4.1 `Deliberating` did not add a useful action

A deliberating driver still had to select one of two possibly contested lanes. It had no additional observation, communication channel, priority rule, or ability to yield. Consequently, deliberation could not resolve the symmetric encounters responsible for the conceptual problem. At most it duplicated the existing avoidance calculation.

Calling this a capability would give architectural form to a mechanism without a causal function.

### 4.2 Generic components do not automatically constitute meaningful composition

Names such as `Memory` or `LocalPerception` are too broad. A valid component must answer:

- What state does it contain?
- What information can update it?
- Which system reads it?
- Which decision can it change?
- What behavioral hypothesis does its presence represent?

Otherwise capability composition is merely a more elaborate taxonomy of agents.

### 4.3 Structural mutation must be theoretically motivated

Adding `Deliberating` during danger and removing it during stability would create archetype changes, but it would not necessarily represent a defensible transition in the reference theory. ECS should make genuine structural change easy; it should not force every change in activation level to become structural.

Stable capabilities can remain components for a driver's lifetime. Continuous variables can represent activation and learning. Components should be added or removed during life only where the model claims that a capability is genuinely acquired or lost.

### 4.4 Mean habitus and imitation risk restoring perfect information

Exact mean habitus is a modeler's statistic over private states. It is not an observable social fact. Similarly, imitation becomes another oracle if drivers copy “successful” agents using information they could not possess. Drivers can observe realized lane use and perhaps survival over time; they cannot observe the internal reasons for it.

### 4.5 Capabilities cannot repair an impoverished action space

With mandatory unit-speed movement, every encounter must be resolved through lane choice. Adding cognitive labels cannot create a genuinely defensive response. The model needs an observable physical state and a new action before local risk assessment can become useful.

## 5. Proposed redesign

### 5.1 Separate four kinds of state

**Universal physical state** describes what exists and can be observed:

- `Position`
- `PrevPosition`
- `Direction`
- `Speed`
- `Age` or `Step`

**Stable capability components** identify mechanisms a driver possesses:

- `SameDirectionResponse(sensitivity)`
- `OppositeDirectionResponse(sensitivity)`
- `NearFieldAvoidance(sensitivity)`
- `HabitFormation(gene)`
- `ConventionPerception(horizon, learning_rate, noise)`
- `SpeedAdjustment(desired_speed, braking_horizon, acceleration_delay)`

**Acquired private state** belongs only to relevant capabilities:

- `Habitus(value)` for entities with `HabitFormation`
- `PerceivedConvention(value, confidence)` for entities with `ConventionPerception`

**Transient decision state** is rewritten each step:

- `LocalObservation`
- `LaneScore`
- `LaneProposal`
- `SpeedProposal`
- `MovementPath`

There is no `DriverStrategy` enum and no shared predicted-occupancy forecast.

### 5.2 Keep lane calculation bounded

The lane systems should retain the interpretable terms of the reference model. Each optional capability contributes to `LaneScore` for the entities that possess it:

```text
same-direction response system  ─┐
opposite-direction system       ├─→ LaneScore → LaneProposal
near-field avoidance system     ┤
habit system                    ┤
convention tie-break system     ─┘
```

All inputs come from current local physical observations, personal acquired state, or lagged observations. No system reads another driver's habit, sensitivities, proposal, or capability bundle.

`PerceivedConvention` is updated from observed realized lane use, expressed relative to each observed driver's direction. It is not the exact mean habitus. It should influence a choice only when direct lane evidence is weak:

```text
if lane-score difference is material
    follow bounded local calculation
elseif personal habit is established
    follow habit
elseif perceived convention is sufficiently clear
    follow perceived convention
else
    retain the current side or make an error-prone choice
```

This makes the convention a correlated, historically produced tie-breaker rather than hidden global knowledge.

### 5.3 Add speed as the genuine local safety mechanism

Use positive integer speed in the range 1–3. Drivers seek the highest locally safe speed, searching from their maximum downward. Speed 1 is therefore the most defensive available action, not a stopped state, and a driver never solves a difficult encounter simply by remaining motionless.

Drivers with `SpeedAdjustment` extrapolate nearby cars under the bounded assumption that currently observed speed and lane persist. For each candidate speed they check both lanes, preferring the lane selected by bounded lane calculation when both support the same speed. They do not predict another driver's lane intention or acceleration and cannot inspect another driver's current proposal.

This addition changes the action set without assuming away coordination. A driver for whom a fast path looks exposed can slow down or select the other lane. Because every driver reasons from lagged observable motion and proposals remain simultaneous, two locally safe choices can still be jointly unsafe. That residual uncertainty preserves a role for convention and habit.

Movement is resolved in three micro-steps. Same-cell, swap, diagonal-crossing, and intermediate-path conflicts are detected at each micro-step, so a speed-3 car cannot pass through another car merely because their final cells differ. The ring length and observation horizon are tripled to 300 and 60 cells; the initial population is tripled to 120, preserving the original 20% road occupancy.

### 5.4 Break symmetry with observable asymmetry and shared history

The requirement is better described as breaking otherwise symmetric encounters. The redesign uses three mechanisms:

1. **Path-dependent personal habit.** Different histories generate different individual defaults.
2. **Lagged perceived convention.** Nearby committed behavior supplies a partially shared signal that can correlate choices without exposing private state.
3. **Observable speed heterogeneity.** Current speed and distance make some local encounters asymmetric and create a meaningful slowing response.

An exact global convention signal can be added as a separate treatment, but only if interpreted as a public institution such as a communicated traffic rule. It should not be called mean habitus.

## 6. Parallel interaction semantics

Parallel execution must mean that agents make decisions from the same committed state, not that they know one another's simultaneous decisions. Use a staged scheduler:

```text
1. Rebuild physical occupancy from committed positions and speeds
2. Compute bounded local observations
3. Update personal convention perceptions from past committed behavior
4. Reset lane scores
5. Apply capability-specific lane contributions
6. Finalize lane proposals
7. Select the fastest locally safe speed and expand proposals into movement paths
8. Detect destination, swap, diagonal, and occupied-cell conflicts
9. Commit all non-conflicting movement simultaneously
10. Update habit from realized action
11. Remove collisions and create replacements
12. Log outcomes
```

Within each stage, systems can process entity batches in parallel because each agent writes only its own observation, score field, or proposal. The conflict stage is a reduction over proposed paths and must be deterministic and independent of query order. No proposal becomes observable until the commit boundary.

This architecture separates three ideas that were previously conflated:

- parallel computation;
- simultaneous behavioral semantics;
- information available to agents.

The engine can compute every proposal in parallel without granting agents contemporaneous access to other proposals.

## 7. Replacement and selection

The reference model requires replacement to renew variation while differential survival changes the population of surviving dispositions. The substantive extension should therefore:

- preserve direction to maintain directional balance;
- place the replacement in a random free position;
- initialize `Speed` from the admissible 1–3 entry distribution;
- initialize habitus and perceived convention without mature private experience;
- sample stable capability components and their parameters from the treatment's entry distribution.

It should not give a replacement the capability bundle of the driver that crashed. Strategy-preserving replacement is appropriate only for a controlled-composition experiment where fixed group sizes are explicitly required.

The implemented `EvolutionaryReplacement` treatment copies a uniformly selected survivor's capability bundle and continuous trait values, then applies independent capability-presence mutations and Gaussian quantitative-trait mutations. It is an explicit alternative to the default entry-draw replacement rather than an implicit change to the baseline. Direction follows the crashed car, while habitus and perceived-convention confidence reset at birth.

## 8. Experimental structure

### Stage A: faithful replication

Reproduce the reference model without the forecast strategies. Establish convention strength, collision/replacement rates, habit effects, and update-order sensitivity.

### Stage B: synchronous replication

Use the same behavioral inputs and equations with staged simultaneous proposals. Treat differences from the sequential model as substantive timing results, not merely performance results.

### Stage C: compositional extension

Vary the shares and combinations of three theoretically distinct capabilities:

1. `HabitFormation`
2. `ConventionPerception`
3. `SpeedAdjustment`

The original traffic-response components can remain the common bounded-calculation baseline. This yields interpretable questions instead of a strategy tournament:

- Can locally perceived convention substitute for personal habit?
- Does speed adjustment reduce selection pressure for convention formation?
- Does habit allow higher throughput by making yielding less necessary?
- Which capability bundles survive when newborn composition is replenished but not fixed?

### Stage D: information treatments

Compare local horizon, observation noise, lag length, and—only as a labelled institutional treatment—a public convention signal. Decision-aware prediction can remain as an oracle bound in an appendix, but it should not be interpreted as bounded behavior.

## 9. Outcomes and falsification criteria

Collision counts alone reward a population that remains stopped. Report:

- collisions or replacements per car-step;
- collisions per distance travelled;
- successful cell movements and throughput;
- mean speed and low-speed share;
- convention strength and time to convention;
- mean absolute habitus and habit persistence;
- accuracy and dispersion of perceived convention;
- survival and population share by capability combination.

The extension is informative even if speed adjustment or convention perception fails. In particular:

- If speed adjustment lowers collisions only by destroying throughput, it is not a useful safety mechanism.
- If exact public convention information is necessary but local perception fails, the institutional information assumption must be made explicit.
- If capability composition does not affect convention formation after controlling for parameters, the claimed scientific benefit of structural heterogeneity is weak.
- If synchronous results differ radically from the sequential replication, scheduling is part of the model and must not be presented as an implementation-neutral optimization.

## 10. Concrete implementation sequence

1. Freeze and document a faithful replication baseline.
2. Remove forecast strategies from the main scientific experiment; retain oracle results separately.
3. Replace `DriverStrategy` with optional model-specific capability components.
4. Split objective occupancy, private acquired state, and transient proposals.
5. Implement the staged synchronous scheduler and private lane proposals.
6. Add `PerceivedConvention` using lagged local realized behavior.
7. Add positive integer `Speed` in 1–3 and `SpeedAdjustment`, followed by swept path-aware conflict detection.
8. Change substantive replacement to redraw capability bundles and reset acquired state.
9. Add deterministic epistemic tests: changing an unobservable trait of another car must not change a focal driver's observation or proposal.
10. Run the factorial capability, density, and information experiments with throughput as well as survival outcomes.

This sequence preserves a clear boundary between replication and extension. It uses ECS to express scientifically meaningful composition, makes interaction simultaneous without making information perfect, and introduces observable asymmetry through a real additional action rather than through a more powerful forecasting algorithm.
