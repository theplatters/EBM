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
  #scale(x: 130%, y: 130%, reflow: true)[#ecs-data-layout]
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
    previous positions $arrow$ occupancy + bounded observations $arrow$ one LR/lane proposal
    $arrow$ speed/path proposal $arrow$ synchronous micro-step conflict resolution
    $arrow$ traces/replacement $arrow$ state and logging
  ]
]

- All cars decide from *one committed pre-decision state*.
- The world resolves their proposed paths together; successful drivers leave
  traces, collided cars are replaced, then acquired state and aggregates are
  updated.
- This ordering makes both update order and information availability explicit.

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

== Acquired capabilities
#figure()[
  #image("../../plots/social_habit_comparison.png", height: 85%)
]
#align(center)[#text(
  size: 13pt,
)[*Takeaway:* among synchronous treatments, SocialHabit is the strongest pure capability; static mixing improves further.]]
#speaker-note[
  The plot is the 30-paired-seed, post-burn-in comparison. SocialHabit's
  advantage reflects successful-driver traces observed locally, not direct
  access to driver success or a population-level statistic.
  The black and purple columns are unit-speed sequential references with
  within-tick ordering information, not like-for-like capability treatments.
]

== Evolutionary mixture
#figure()[
  #image("../../plots/social_habit_mixture_ensemble_dynamics.png", height: 56%)
]
#align(center)[#text(size: 13pt)[*Takeaway:* evolution improves coordination without collapsing profiles.]]
#align(center)[#text(
  size: 12pt,
)[Convention *0.867 [0.848, 0.885]*  ·  coordinated time *0.828 [0.797, 0.859]*  ·  replacements/car-step *0.0380 [0.0357, 0.0403]*]]

= Possible benefits and drawbacks of using ECS for ABMs

== Benefits
- More natural support for *parallel agent interactions*, both conceptually and computationally.
- *Compositional heterogeneity* offers a new approach to heterogeneous agents. (Dynamic acquisition of new capabilities)
- Potential *performance* benefits compared to "agent-oriented" implementations @JuliaDynamicsABMFrameworksComparison2026.
  - Easy to utilize the GPU.
  - More ergonomic than the big parallel ABM frameworks (FLAME GPU, ) with potentially minimal loss in performance.
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



== Single-path diagnostics

#figure(caption: [Analytical histories expose temporal variation in a current capability run.])[
  #image("../../plots/capability_dynamics.png", height: 52%)
]
#align(center)[#text(
  size: 12pt,
)[*Diagnostic:* seed `20260730`, mixed static entry; one path illustrates fluctuation, not replicated evidence.]]

== Lane-first vs. speed-first
#figure()[
  #image("../../plots/speed_sensitivity_comparison.png", height: 65%)
]
#align(center)[#text(size: 13pt)[*Takeaway:* lane-first improves progress and reduces replacements under both regimes.]]
#speaker-note[
  Lane-first raises completed progress by 0.166 [0.154, 0.178] cells per
  car-step under static entry and 0.049 [0.040, 0.058] under evolution;
  replacements per car-step fall by 0.0486 and 0.0162. Clearance and a speed-2 cap are
  different treatments, not generic improvements. The published baseline
  remains the explicit historical speed-first rule.
]

== Lane-first capability comparison
#figure()[
  #image("../../plots/lane_first/social_habit_comparison.png", height: 60%)
]
#align(center)[#text(
  size: 13pt,
)[*Takeaway:* lane-first transforms synchronous coordination — every capability treatment now matches or beats the sequential reference.]]
#speaker-note[
  Full rerun of the published 5,000-tick, 30-paired-seed design with
  `prefer_lane_over_speed=true`. Convention strength rises from 0.439 to 0.874
  (Habit), 0.429 to 0.769 (Convention), 0.508 to 0.885 (SocialHabit), 0.630 to
  0.911 (static entry), and 0.867 to 0.944 (evolutionary); replacement rates
  fall by roughly three quarters. Even no-habit improves slightly
  (0.230 $->$ 0.262). Lane-first only reorders the action search of the
  synchronous capability model, so the sequential reference rows are
  unchanged.
]

== Lane-first mixture dynamics
#figure()[
  #image("../../plots/lane_first/social_habit_mixture_ensemble_dynamics.png", height: 62%)
]
#align(center)[#text(
  size: 13pt,
)[*Takeaway:* under lane-first, coordination is faster and more stable; evolution still adds robustness.]]
#speaker-note[
  Static-entry post-burn-in convention is 0.910 with replacement pressure
  0.0311; evolutionary is 0.944 with 0.0230. Coordinated time reaches 0.932
  (static) and 0.982 (evolutionary). Capability-profile diversity remains
  dispersed under evolution, mirroring the speed-first result.
]

== Lane-first diagnostics
#grid(
  columns: (1fr, 1fr),
  column-gutter: 1em,
  [
    #figure(caption: [No-convention ablation (5 paired seeds).])[
      #image("../../plots/lane_first/no_convention_comparison.png", height: 52%)
    ]
  ],
  [
    #figure(caption: [Static-entry single-seed history.])[
      #image("../../plots/lane_first/capability_dynamics.png", height: 52%)
    ]
  ],
)
#align(center)[#text(
  size: 12pt,
)[*Diagnostic:* habit's coordination benefit survives without convention learning; single-seed paths still fluctuate.]]

== Robustness and interpretation

- The capability ensemble samples histories every 25 ticks and marks the
  1,000-tick burn-in; broad regime averages are stable after burn-in while
  individual windows fluctuate.
- The sequential reference is not like-for-like with synchronous capability
  outcomes: it is unit-speed and supplies within-tick activation-order
  information.


== Bibliography

#bibliography("../Econ.bib")
