#import "@preview/touying:0.6.3": *
#import themes.university: *
#import "@preview/numbly:0.1.0": numbly
#import "@preview/theorion:0.4.1": *
#import cosmos.clouds: *
#import "diagrams.typ": classic-abm-data-layout, ecs-data-layout, traditional-abm-layout, what-if-layout
#show: show-theorion
#let info-box(title: "Info", body) = {
  block(
    fill: blue.lighten(90%),
    stroke: (left: 4pt + blue),
    inset: 1em,
    radius: 4pt,
    width: 100%,
    stack(
      spacing: 0.5em,
      text(weight: "bold", title),
      body,
    ),
  )
}

#set cite(style: "apa")

#show: university-theme.with(
  aspect-ratio: "16-9",
  config-common(frozen-counters: (theorem-counter,)), // freeze theorem counter for animation
  config-info(
    title: [Rethinking ABM Architecture],
    subtitle: [Entity Component Systems and the Case for Parallel Agent Interaction],
    author: [Franz Scharnreitner],
    date: datetime(year: 2026, day: 04, month: 03),
    institution: [ICAE Linz],
  ),
)

#set heading(numbering: numbly("{1}.", default: "1.1"))

#title-slide()


= Cars driving in circles

== The original model

- Influential paper by #cite(<hodgsonEconomicsShadowsDarwin2006>, form: "prose") #pause
#align(center)[

  #image("assets/roundabout.jpg", height: 70%)
]
== Overview
- Cars move sequentially, either clockwise or counterclockwise, on a $100 times 2$ ring.
- In each step, a car chooses the left or right lane according to the decision formula
  $ "LR"^n = s^n + o^n + c^n + h^n $.
- Here, $s^n$, $o^n$, and $c^n$ depend on the cars ahead of car $n$, while $h^n$ represents the driver's habit.
- Driving on one lane shifts habit toward that lane, so $h^n = h^n_"prev" + "lane" / (K + t^n)$.


== Main results of Hodgson & Knudsen

- *Habit is crucial for the emergence of convention.*
- Repeated lane choices gradually become ingrained, making left/right coordination more stable.
- Agents do not coordinate only because of immediate incentives; they also coordinate because past behavior shapes current disposition.
- Habit therefore strengthens and stabilizes conventions beyond what purely reactive decision-making can achieve.
- The paper’s broader claim is that institutions persist partly because they become internalized as habits.

== The real world

#speaker-note[
  + Ask what information an activation order gives later drivers within the same tick.
  + Emphasize that update timing is a modeling choice shaped, but not determined, by architecture.
  + The timing slide defends joint resolution as a behavioral claim about observability, not a clock claim.
]
- Drivers on a roundabout decide on the basis of locally observable cues: available gaps, speeds, brake lights, and lateral positions.
- Decisions are taken continuously and concurrently, without a global clock or fixed turn order @hubermanEvolutionaryGamesComputer1993.
- The intentions of other drivers remain unobserved until expressed as visible motion. #pause
- To model this: all proposals from one observable state, then joint resolution
- What is the problem when modeling this in an "agent-centered framework"?

= ABM layouts

== Traditional ABM layout
#align(center)[#traditional-abm-layout]
#speaker-note[
  + Agents have step functions
  + these step functions run sequentially
]

== What if?

#speaker-note[
  + What if we group the data not by agents but by systems
  + Luckily  paradigm exists => ECS
]
#align(center)[#what-if-layout]


= ECS (Entity Component System)!


== The history of ECS

#grid(
  columns: (1fr, auto),
  column-gutter: 1.2em,

  [
    - The origins of ECS reach back to 1959, when Ivan Sutherland pioneered a similar idea in a drawing program, one of the first graphical user interfaces @sutherlandSketchpadManmachineGraphical2003.
    - Today, ECS is primarily used in game development, for example in the Bevy engine and Unity DOTS.
    - While there is a paper on using ECS for ABMs, it remains a largely unknown programming paradigm in this area @casalsHECATEECSbasedFramework2025.
  ],

  [
    #image("assets/sketchpad.png", width: 180pt)
  ],
)


== A short introduction


- ECS stands for *Entity Component System*. It is a programming pattern that separates data from behavior.
- *Entities* are unique identifiers. By themselves they do not contain logic or meaning; they simply represent individual objects in the simulation.
- *Components* are small data containers attached to entities. Each component stores one specific aspect of state, such as position, direction, age, or velocity.
- An entity is defined by the set of components it has. For example, a car entity might have components for position, direction, habitus, and step count.
- *Systems* implement the behavior of the model. They query all entities that have a required set of components and then update those components according to the model rules.
- This means that data and functionality are kept separate: components hold data, while systems contain the logic that acts on that data.
#pagebreak(weak: true)


== Data layout in a classic ABM


#classic-abm-data-layout
== ECS data layout

#v(0.5fr)
#align(center)[
  #scale(x: 150%, y: 150%, reflow: true)[#ecs-data-layout]
]

#v(0.5fr)

#speaker-note[
  + Contrast with the previous slide: the table is transposed — state lives in component rows, not inside agent objects.
  + Dashed cells are capabilities a car has not acquired; which systems apply to a car follows from its component set.
  + Each arrow is a query, so what a system reads and writes — its information availability — is explicit in the structure.
]

== ABM in ECS terms
#table(
  columns: (auto, auto, auto, auto),
  inset: 7pt,
  align: horizon,
  [*Concept*], [*Description*], [*ABM Equivalent*], [*Role in Simulation*],

  [*Entity*], [Unique object in the world], [Agent], [Represents an individual actor in the simulation],

  [*Component*],
  [Data attached to an entity],
  [Agent attributes / state variables],
  [Stores properties such as position, preferences, or resources],

  [*System*],
  [Function operating on sets of components],
  [Part of the agent step function],
  [Implements simulation rules and updates state],
)

== The habit rule in both layouts

#v(2.0cm)
#grid(
  columns: (1.05fr, 1.15fr),
  column-gutter: 1.5em,
  [
    #text(size: 11pt)[
      *Classic ABM (Agents.jl)* \
    ]
    #align(left)[
      #text(size: 10.0pt)[
        ```julia
        @agent struct Car(GridAgent{2})
            lr::Float64
            habitus::Float64
            direction::Direction
            age::Int64
        end

        function update_habitus!(agent, model)
            agent.habitus = clamp(
                agent.habitus + relative_lane_sign(
                    agent.pos[1], agent.direction
                ) / (model.params.K + agent.age),
                -1.0, 1.0,
            )
        end

        function agent_step!(agent, model)
            calculate_lr!(agent, model)
            move_agent!(agent, intended_position(agent, model), model)
            agent.age += 1
            update_habitus!(agent, model)
        end
        ```
      ]
    ]
  ],
  [
    #text(size: 11pt)[
      *ECS (Ark.jl)* \
    ]
    #align(left)[
      #text(size: 10.0pt)[
        ```julia
        struct Position
            x::Int64
            y::Int64
        end
        struct Habitus
            val::Float64
        end

        function update_habitus!(world)
            params = Ark.get_resource(world, ModelParams)
            for (e, pos, dir, hab, step) in
                Query(world, (Position, Direction, Habitus, Step))
                @inbounds for i in eachindex(e)
                    hab[i] = Habitus(clamp(
                        hab[i].val + relative_lane_sign(
                            pos[i].x, dir[i]
                        ) / (params.K + step[i].val),
                        -1.0, 1.0,
                    ))
                end
            end
        end
        ```
      ]
    ]
  ],
)

#speaker-note[
  Both snippets implement the same Hodgson--Knudsen habit rule on the same state. In the classic layout, the habit update is one line inside the agent's step function: behavior is bundled per car, and the framework runs these steps agent by agent, inviting sequential updates. In ECS the same rule is a system: a bulk operation over a query of all matching cars. Update order, information availability, and parallelizability become explicit structural choices.
]

= Current traffic model


== Capability tick

#align(center)[
  #text(size: 18pt)[
    committed positions $arrow$ bounded observations $arrow$ binding LR/lane decision
    $arrow$ risk-adjusted speed proposal $arrow$ synchronous micro-step collision detection
    $arrow$ successful-driver traces/replacement $arrow$ acquired state/logging
  ]
]

- All cars decide from *one committed pre-decision state*.
- LR determines the final lane; speed is chosen only on that binding lane.
- A car attempts maximum speed (normally 3), makes one seeded draw, and accepts danger with probability $1 - "risk aversion"$.
- Rejected danger causes progressively lower-speed tests; speed 1 is the unavoidable-risk fallback when every speed is dangerous.
- Successful drivers leave traces, collided cars are replaced, then acquired state and aggregates are updated.

== What enters LR?

- *Habit:* the driver's own realized-side history, reinforced by the age-dependent Hodgson--Knudsen rule.
- *Convention:* a private history of locally observed side choices made by *other* drivers @ellisonLearningLocalInteraction1993.
- *SocialHabit:* a private history of locally observed, decaying traces left by successful drivers.

LR is an *additive score*: each mechanism adds its own weighted term, $w_h dot "disposition" dot h^n$ (habit), $w_c dot "confidence" dot c^n$ (convention), via a dedicated scoring system; `propose_lanes!` commits the *sign* of the sum.


== Timing and habit
#figure()[
  #image("../../plots/activation_habit_results.png", height: 85%)
]
#align(center)[#text(
  size: 13pt,
)[*Takeaway:* timing changes compatibility; habit more than offsets the simultaneous loss.]]
#speaker-note[
  This experiment uses SequentialModel's activation-order and explicit
  simultaneous schedulers; it isolates timing semantics rather than switching
  the CapabilityModel's update rule. The timing sensitivity echoes the
  classical synchronous-versus-asynchronous results for spatial games
  @hubermanEvolutionaryGamesComputer1993 @newthAsynchronousSpatialEvolutionary2009a,
  and the direction agrees: staggered information buys coordination.
  Their asynchronous prescription targets systems without any shared rhythm,
  whereas road traffic has one: drivers react to what is visibly happening
  around them, not to a private schedule. They still decide alone, each
  against the same visible state; at this timescale a decision responds to
  the road, not to a neighbor's uncommitted intention. Modeling the period
  as simultaneous decisions is therefore a behavioral claim, not a clock
  artifact, and the architecture makes it explicit.
  The simultaneous/no-habit effect is -0.00312, with 95% bootstrap interval
  [-0.00351, -0.00272] (magnitude 0.00272--0.00351). Habit raises simultaneous
  compatibility by 0.00474 [0.00433, 0.00512]. This is an encounter-level
  result; the aligned one-tick fixture is a mechanism check, not evidence that
  alignment emerges by itself.
  Activation order here is fixed by car ID; per-tick shuffling would
  redistribute the early-mover advantage across drivers without removing
  the underlying leakage.
]

== Capabilities under uniform entry risk
#figure()[
  #image("../../plots/uniform_risk_capability_comparison.png", height: 85%)
]
#align(center)[#text(
  size: 13pt,
)[*Takeaway:* non-heritable entry risk lets the mixed evolutionary treatment isolate capability evolution from inherited-risk selection.]]
#speaker-note[
  30 paired seeds 20260901:20260930, 5,000 ticks, 1,000 burn-in, population 120,
  a 2 x 300 ring, lookahead 20. Six synchronous CapabilityModel treatments and
  two sequential references. Five synchronous treatments use EntryDrawReplacement,
  so capabilities are redrawn from entry shares and risk is independently
  Uniform(0,1) on replacement with no evolution. The mixed evolutionary treatment
  uses EvolutionaryReplacement, so stable capabilities and quantitative traits
  inherit and mutate, but an experiment-only post-step hook replaces every
  newborn's temporarily inherited RiskAversion with an independent seeded
  Uniform(0,1) draw before its next decision; that risk stays fixed for the car's
  lifetime and is non-heritable. Capability mutation rate is .02 and quantitative
  trait mutation scale is .05. Sequential references have no RiskAversion
  component. Keeping risk independent and non-heritable separates capability
  evolution from inherited-risk selection.
]

== Mixed capability dynamics
#figure()[
  #image("../../plots/uniform_risk_mixture_ensemble_dynamics.png", width: 88%)
]
#align(center)[#text(
  size: 13pt,
)[*Takeaway:* mean capability shares are higher under mixed evolution, but between-seed variation is substantial.]]
#speaker-note[
  This dynamics artifact comes from `notebooks/run_mixture_ensemble_dynamics.jl`.
  Design: 30 paired seeds 20260901:20260930, 5,000 ticks, 1,000-tick burn-in,
  sampling every 25 ticks, population 120 on a 2 x 300 ring, lookahead 20. Two
  scenarios run side by side — a static mixture using `EntryDrawReplacement`, and
  an evolutionary mixture using `EvolutionaryReplacement`. In the static scenario
  capabilities are redrawn from entry shares with no evolution; in the evolutionary
  scenario capabilities and quantitative traits inherit and mutate, but after each
  replacement a seeded, independent Uniform(0,1) draw overwrites the newborn's
  temporarily inherited risk before its next decision, so risk is fixed for the
  car's life and non-heritable. Solid lines are across-seed means and the ribbons
  are ±1 SD; the dashed line marks the burn-in boundary. Mean final shares match
  the uniform-risk comparison CSV: static Habit .513, Convention .514,
  SocialHabit .515, and all three .133, versus evolutionary .577, .563, .600,
  and .179. The gap is real in sample means, but the evolutionary trajectories
  show substantial between-seed variation, so the figure supports a difference in
  ensemble means rather than a universal directional claim for every seed.
]

= Possible benefits and drawbacks of using ECS for ABMs

== Benefits
- More natural support for *parallel agent interactions*, both conceptually and computationally.
- *Compositional heterogeneity* offers a new approach to heterogeneous agents. (Dynamic acquisition of new capabilities)
- Potential *performance* benefits compared to "agent-oriented" implementations @JuliaDynamicsABMFrameworksComparison2026.
  - Easy to utilize the GPU.
  - More ergonomic than the big parallel ABM frameworks (FLAME GPU) with potentially minimal loss in performance.
- Clear support for *modularity* and separation of concerns
- Encourages a more *systemic* rather than purely individual-centered modeling perspective

== Drawbacks

- *Agent-centric thinking does not map directly onto entity*--component tables;
  the mental shift is substantial for most modelers.
- *Boilerplate*: every state variable becomes a wrapper type, and behavior is
  scattered across many small systems.
- Very *limited ABM literature* and tooling: no equivalent of the Agents.jl,
  Mesa, or NetLogo ecosystems @casalsHECATEECSbasedFramework2025.
- Performance is framework- and workload-dependent
  @JuliaDynamicsABMFrameworksComparison2026.

#speaker-note[
  These costs are visible in the traffic codebase: replacement requests are
  staged between systems and movement snapshots are sorted explicitly for
  reproducibility, and lane scores, observations, and dispositions each live
  in their own singleton structs. None of this was fatal, but it raises the
  entry floor relative to writing one `agent_step!` function.
]

== Outlook
- Scheduler (Helm.jl) for Ark.jl in progress.
  - Helps make dependencies of systems explicit, leading to automatic parallel scheduling between systems and easier modularity.
  - Could enable automatic ODD exporting @grimmODDProtocolDescribing2020.
- Rewrote BeforeIT @glielmoBeforeITjlHighPerformanceAgentBased2025 @polednaEconomicForecastingAgentbased2023 from a Structure of Arrays approach to ECS
  - Modular Macro-ABM?
- More experiments with ABMs that truly benefit from compositional heterogeneity.


== Conclusion
- *The choice of framework is not neutral:* it affects modeling decisions
  about the information agents hold, update ordering, sequential versus
  parallel action, and the type of heterogeneity agents possess.
- *ECS offers an alternative to agent-oriented frameworks*.
- The extensions of #cite(<hodgsonEconomicsShadowsDarwin2006>, form: "prose") show how architecture
  shapes which modeling choices are easiest to express: agent-oriented layouts
  favor activation-order updates, while ECS makes staged synchronous resolution
  more explicit
- *ECS is promising for ABMs, but remains underexplored* conceptually and
  methodologically.
#show: appendix

= Appendix



== Robustness: fixed and evolving risk aversion
#figure()[
  #image("../../plots/risk_aversion_comparison.png", height: 85%)
]
#align(center)[#text(
  size: 13pt,
)[*Takeaway:* fixed zero risk has the highest throughput, while replacement pressure is nonmonotone.]]
#speaker-note[
  This is the current 30-paired-seed risk experiment: seeds 20260901:20260930,
  5,000 ticks, 1,000-tick burn-in, 120 cars, a 2 x 300 road, lookahead 20,
  mixed capabilities, and mutation rate 0.02.
]

== Robustness: risk distributions
#figure()[
  #image("../../plots/risk_aversion_distribution.png", height: 70%)
]
#align(center)[#text(
  size: 13pt,
)[*Takeaway:* evolution improves average outcomes, but survival selection is path dependent.]]
#speaker-note[
  Entry risk is Uniform(0, 1). Evolution uniformly selects a survivor and
  inherits risk and capabilities with Gaussian mutation scale 0.05, clamped to
  [0, 1]. Fixed interventions are reset before each decision without consuming
  RNG. The evolved final mean is below .5, but huge between-seed variation and
  low/high-risk regimes prevent a claim of universal selection toward lower risk.
]


== Bibliography

#bibliography("../Econ.bib")
