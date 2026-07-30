#import "@preview/ilm:1.4.2": *

#set text(lang: "en")

#let scaffold-note(title, body) = block(
  width: 100%,
  inset: 8pt,
  radius: 3pt,
  fill: luma(246),
  stroke: luma(215),
)[
  #text(weight: "bold")[#title]
  #v(0.25em)
  #body
]

#show: ilm.with(
  title: [Beyond the Agent Object: Entity Component Systems as an Architecture for Agent-Based Modeling],
  author: ("Franz Scharnreitner BSc."),
  date: datetime(year: 2026, month: 7, day: 25),
  abstract: [
    Agent-based models are commonly implemented around agent objects that combine identity, state, and behavior. Although this organization mirrors the intuitive description of autonomous agents, it makes predefined agent types the principal unit of model construction and can obscure the population-level processes through which agents interact. This paper examines Entity Component Systems (ECS) as an alternative architecture for agent-based modeling. ECS represents agents as dynamically composed sets of state components and behavior as systems operating on all entities with the required capabilities. We develop three connected arguments: ECS supports structural rather than merely parametric heterogeneity; it aligns executable model structure with population-level processes; and it exposes the data dependencies required for simultaneous and parallel interaction. These arguments are developed through an ECS reconstruction and extension of Hodgson and Knudsen's traffic-convention model and a comparison with an agent-centered implementation. A secondary engineering evaluation considers cache locality, multicore scaling, and the prospects for GPU execution. The paper positions ECS not only as a computational optimization, but as an alternative way of specifying agent-based models.
  ],
  bibliography: bibliography("Econ.bib"),
  figure-index: (enabled: true),
  table-index: (enabled: true),
  listing-index: (enabled: true),
)

#scaffold-note([Status of this document], [
  This is a manuscript scaffold. Each section records the claim it should establish, the evidence it requires, and important qualifications. Replace these notes with finished prose only after the corresponding implementation or analysis exists.
])

= Introduction

Agent-based modeling (ABM) is used to explain macro-level regularities as the emergent result of heterogeneous agents and their interactions. In most implementations, the scientific agent is mapped onto a software object that combines identity, state, and behavior. This mapping is intuitive, but it is not neutral: it encourages modelers to begin with a taxonomy of agent types, locate behavior in agent-local step functions, and treat population-level processes as consequences of repeatedly invoking those functions.

Entity Component Systems offer a different starting point. Entities supply identity, components represent state and capabilities, and systems implement transformations over every entity possessing a required component signature. The resulting architecture is organized around composition and processes rather than class membership and agent-local methods.

This paper asks whether that architectural change is useful for scientific ABMs, not merely whether an ECS engine can execute an existing model faster.

== Motivation

- ABMs value heterogeneity, yet software implementations commonly distinguish agents through parameter values or predefined types.
- Interactions are often population-level processes even when their implementation is distributed across agent methods.
- Simultaneous interactions require a distinction between observation, intention, conflict resolution, and commitment.
- Large simulations repeatedly apply homogeneous operations to selected state variables, making memory layout and batch execution consequential.

== Research questions

+ *RQ1 -- Compositional heterogeneity:* How does ECS change the representation of agents with overlapping and dynamically changing capabilities?
+ *RQ2 -- System-centered modeling:* What is gained and lost when behavior is organized as population-level systems rather than agent-local methods?
+ *RQ3 -- Agent interaction:* Does ECS provide a more direct formulation of simultaneous and parallel interactions while keeping their semantics explicit?
+ *RQ4 -- Engineering consequences:* Under which workloads do ECS data layouts improve cache use, multicore scaling, or GPU suitability?

== Intended contributions

+ A distinction between parametric, type-based, and compositional heterogeneity in ABM implementations.
+ A formal mapping from ABM concepts to entities, components, queries, systems, and resources.
+ A comparative reconstruction of a classical economic ABM in agent-centered and ECS architectures.
+ An ECS extension in which behavioral capabilities overlap and can change without defining new agent classes.
+ A staged formulation of simultaneous interactions derived from system dependencies.
+ A measured, qualified assessment of cache behavior and parallel scaling.

#scaffold-note([Introduction still needs], [
  Open with one concrete example in which adding a behavioral capability or changing interaction semantics forces a cross-cutting change in an agent-centered model but only a local change in the ECS model. End the finished introduction with a concise statement of results, not just the research questions.
])

= Agent-Centered Architectures and Their Commitments

== The agent object convention

Define the representative architecture under comparison:

- An agent has a named type.
- Its state is stored in fields belonging to that type.
- Its behavior is implemented by one or more methods, commonly a step or event-handler function.
- A scheduler selects agents and invokes those methods.
- Interactions are performed through direct access to other agents or a shared model state.

The relevant contrast is *agent-centered versus system-centered architecture*, not Julia versus another language and not an assertion that object-oriented programming is incapable of composition. Existing ABM frameworks are diverse and often offer custom activation, multiple agent types, and event-based execution @abarAgentBasedModelling2017.

== Three kinds of heterogeneity

#table(
  columns: (1.1fr, 1.8fr, 1.8fr),
  inset: 6pt,
  align: (left, left, left),
  table.header([*Form*], [*Representation*], [*Example*]),
  [Parametric], [A common state structure with different values], [Drivers differ in sensitivity coefficients],
  [Type-based], [Membership in predefined nominal classes], [HabitualDriver and CalculativeDriver],
  [Compositional], [Possession of overlapping component sets], [An entity participates in habit, avoidance, or imitation systems according to its current capabilities],
)

The paper must not present heterogeneity itself as an ECS innovation. Its claim concerns *structural and dynamic composition*: capabilities need not correspond to an exhaustive taxonomy of agent classes and may be acquired or removed during a run.

== Behavior located in agents

Introduce a generic agent-centered transition:

$ a_i(t + 1) = f_(tau(i))(a_i(t), N_i(W_t), W_t), $

where $tau(i)$ denotes a predefined agent type, $N_i$ the information observed by agent $i$, and $W_t$ the world state. Explain how a complex step function can bundle perception, decision, action, interaction resolution, and learning even when these processes have different information and timing requirements.

== What this architecture does well

- It closely follows ordinary-language descriptions of autonomous individuals.
- The complete state and behavior of one agent can be inspected in one place.
- Highly individualized cognition or event-driven behavior can be natural to express.
- Mature ABM frameworks provide extensive tooling around this representation.

== Limitations to investigate rather than assume

- Growth of type hierarchies or conditional behavior as capabilities overlap.
- Duplication when several agent types share a process.
- Difficulty separating observation from mutation when simultaneous behavior is required.
- Pointer-heavy or array-of-structures layouts that load irrelevant state during population sweeps.

#scaffold-note([Required standard of comparison], [
  Use an idiomatic agent-centered implementation, not a deliberately weak inheritance hierarchy. Julia is not a conventional class-based OOP language, so describe the Agents.jl implementation as agent-centered. If the paper makes stronger claims about OOP, add a representative class-based implementation or narrow the terminology.
])

= Entity Component Systems as a Modeling Architecture

Existing work has established that ECS can be used to implement ABM engines and can improve parallel performance @ambrosioImpactECSLogic. HECATE applies ECS concepts to engineering multi-agent systems @casalsHECATEECSbasedFramework2025. This paper builds on that literature but shifts the emphasis from engine feasibility to the consequences of ECS for scientific model construction.

== Entities, components, systems, and resources

- *Entity:* an identifier with no intrinsic data or behavior.
- *Component:* a focused unit of state associated with an entity.
- *Component signature:* the set of components currently associated with an entity.
- *Query:* a selection of all entities possessing a required signature.
- *System:* a transformation over components selected by one or more queries.
- *Resource:* model-level state such as parameters, random-number streams, spatial indexes, or aggregate statistics.
- *Structural change:* the addition or removal of entities or components, normally committed at a controlled boundary.

Represent a system as

$ S_k = (Q_k, R_k, W_k, F_k), $

where $Q_k$ selects participating entities, $R_k$ and $W_k$ identify the components or resources read and written, and $F_k$ is the transformation. This representation connects model semantics, dependency analysis, testing, and possible parallel execution.

== Compositional heterogeneity

Let $C(e)$ be the component signature of entity $e$. A system applies when its required signature is present:

$ E_k = { e | Q_k subset.eq C(e) }. $

Develop the following argument:

- An entity does not need to belong to a named class to participate in a behavior.
- Different systems may select overlapping populations.
- A new combination of capabilities does not necessarily require a new agent type.
- Adding or removing a component can represent endogenous acquisition or loss of a capability.
- Heterogeneity becomes a property of component incidence as well as parameter values.

=== Example capability matrix

#table(
  columns: (1.5fr, 0.8fr, 0.8fr, 0.8fr, 0.8fr),
  inset: 6pt,
  align: center,
  table.header([*Entity*], [*Move*], [*Avoid*], [*Calculate*], [*Habit*]),
  [Driver A], [$checkmark$], [$checkmark$], [], [],
  [Driver B], [$checkmark$], [], [$checkmark$], [$checkmark$],
  [Driver C], [$checkmark$], [$checkmark$], [$checkmark$], [$checkmark$],
  [Driver D], [$checkmark$], [], [], [$checkmark$],
)

#scaffold-note([Case-study requirement], [
  The current traffic model gives nearly every car the same component signature and therefore demonstrates mainly parametric heterogeneity. Add an experimental variant with optional and dynamically changing behavioral components before making a strong claim about compositional heterogeneity.
])

== Thinking in systems rather than agents

The agent-centered question is, "What does this agent do during its step?" The system-centered question is, "What transformation occurs in the model, and which entities possess the state required to participate?"

A system-centered transition can be written as

$ S_k: E_k times W arrow Delta W. $

Develop three implications:

+ *Processes become explicit.* Perception, choice, movement, matching, learning, entry, and exit appear as separate executable model processes.
+ *Mechanisms become replaceable.* A modeler can substitute a prediction or learning system while retaining entity state and unrelated processes.
+ *Dependencies become inspectable.* Read and write sets identify which systems must be ordered and which may run independently.

This chapter must address a likely objection: locating behavior outside an entity does not remove scientific agency. Private information, goals, memory, and decisions remain entity-specific components; a system implements the transition law shared by entities possessing the relevant capabilities.

== Interaction phases and concurrency

Distinguish carefully:

- *Simultaneous model semantics:* agents decide from a common state.
- *Concurrent execution:* computations may overlap without changing the declared semantics.
- *Parallel execution:* concurrent work is distributed across hardware for speed.

The ECS decomposition of an interaction should be illustrated as:

```text
observe -> form intentions -> resolve conflicts -> commit actions -> learn
```

Systems with disjoint writes or read-only shared inputs may be evaluated concurrently. Systems that contend over shared resources require an explicit reduction, arbitration, or commitment phase. ECS exposes these dependencies but does not eliminate them. The formal concurrency properties of ECS and the conditions for deterministic execution should be connected to @redmondExploringTheoryPractice2025.

== Data-oriented engineering consequences

Explain the difference between an array of complete agent records and component-oriented storage. Then motivate, without assuming, the following expected effects:

- Systems load only the component arrays they use.
- Homogeneous loops may improve cache locality and vectorization.
- Queries over archetypes can batch entities with compatible memory layouts.
- Explicit read/write sets can support safe multicore scheduling.
- Structure-of-arrays layouts may map naturally to GPU kernels.

Also state the costs:

- Query and archetype-management overhead.
- Expensive structural changes when entities move between archetypes.
- Synchronization and conflict-resolution costs.
- Possible fragmentation of the conceptual description of an individual agent.
- Poor GPU utilization for branch-heavy or tightly coupled interactions.

= Positioning in the Literature

== ABM tools and agent-centered design

- Review major ABM toolkits and how they represent agent types, activation, interaction, and model-level processes @abarAgentBasedModelling2017.
- Avoid claiming that existing frameworks cannot support heterogeneity, synchronous behavior, or custom scheduling.
- Identify the more precise contrast: ECS makes component composition and population systems the default organizing abstractions.

== ECS, data-oriented design, and concurrency

- Introduce ECS origins and its emphasis on composition over inheritance.
- Discuss formal work on ECS semantics and deterministic concurrency @redmondExploringTheoryPractice2025.
- Separate the architectural pattern from particular archetype storage implementations.

== ECS in ABM and multi-agent systems

- Present ECS-based ABM engine work, especially its feasibility and CPU-parallel performance emphasis @ambrosioImpactECSLogic.
- Present HECATE's mapping between ECS and multi-agent engineering @casalsHECATEECSbasedFramework2025.
- Review GPU ABM frameworks whose agent functions, execution layers, messages, or kernels already resemble population-level systems.

== Proposed gap

The intended novelty claim is:

#quote[
  Previous work has established the feasibility and performance potential of ECS-based ABM engines. This paper instead examines ECS as a scientific modeling architecture, focusing on structural heterogeneity through dynamic component composition, the representation of behavior as population-level systems, and the resulting formulation of simultaneous agent interactions.
]

#scaffold-note([Literature work still required], [
  Conduct a reproducible search across ABM, individual-based modeling, multi-agent systems, component-based simulation, process-oriented simulation, FLAME/FLAME GPU, and data-oriented scientific computing. Do not use "first" or "previously unexplored" until this search is documented. Add literature on ODD model descriptions, activation regimes, component-based ABM, and GPU conflict resolution.
])

= Case Study: The Evolution of a Traffic Convention

== Why this model

Use the traffic-convention model associated with Hodgson and Knudsen as a compact economic ABM in which heterogeneous behavioral dispositions, endogenous habit formation, interaction, and convention emergence are tightly connected @hodgsonEconomicsShadowsDarwin2006.

The case study should serve four purposes:

+ Validate that ECS can faithfully express an established model.
+ Make the difference between agent-centered and system-centered organization concrete.
+ Extend the model from parametric to compositional heterogeneity.
+ Demonstrate how the same decomposition supports simultaneous interaction.

== Reference model and replication target

Document using an ODD-compatible structure:

- Purpose and substantive interpretation.
- Agents, environment, spatial scale, and time step.
- Agent state variables and parameter distributions.
- Observation and lane-choice equation.
- Movement and collision rules.
- Habit formation and downward causation.
- Birth, death, or replacement mechanism.
- Initialization and random-number use.
- Original reported patterns to be replicated.

#scaffold-note([Validation requirement], [
  Matched initialization and deterministic tests are necessary but do not establish behavioral equivalence. Define replication targets from the published model, reproduce them with uncertainty, and explain every intentional deviation.
])

== Agent-centered implementation

Describe the faithful port in which a car stores its complete state and executes an agent-local transition. Provide short pseudocode:

```text
for agent in scheduled_agents
    observe current model state
    calculate lane preference
    move immediately
    update habit
end
resolve remaining interactions
```

Record which parts of the transition must change when observation and action are separated or when a new behavioral capability is introduced.

== ECS implementation

#table(
  columns: (1.2fr, 1.6fr, 1.8fr),
  inset: 6pt,
  align: (left, left, left),
  table.header([*Model concept*], [*ECS representation*], [*Responsible process*]),
  [Car identity], [Entity], [Entity lifecycle],
  [Location], [`Position` component], [Movement and occupancy systems],
  [Direction], [`Direction` component], [Observation and movement systems],
  [Sensitivities], [Separate trait components], [Decision systems],
  [Learned disposition], [`Habitus` component], [Habit system],
  [Lane intention], [`LR` or intention component], [Decision and commitment systems],
  [Shared road state], [Occupancy resource], [Index-rebuilding system],
  [Collision], [Derived interaction], [Conflict-resolution system],
  [Replacement], [Deferred entity commands], [Spawning system],
)

Explain the execution graph and identify every system's reads, writes, and structural changes.

== Extension to compositional drivers

Construct a factorial or sampled population of capabilities:

- Perceptual capability: same-direction and opposite-direction traffic sensitivity.
- Calculative capability: explicit forecast-based lane evaluation.
- Avoidance capability: local collision avoidance.
- Habit capability: acquired and reinforced lane disposition.
- Optional imitation or convention-sensitivity capability.

Possible dynamic transitions:

- Add `Habitus` after repeated action establishes a disposition.
- Remove or deactivate costly calculation after behavior stabilizes.
- Activate avoidance following a near collision.

The scientific question is whether convention formation depends on the distribution and co-occurrence of capabilities, not simply on continuous coefficient variation.

== From sequential action to simultaneous interaction

Show how the ECS model can be reorganized without redefining the agent:

```text
rebuild observation state
        |
calculate all lane intentions
        |
calculate all proposed movements
        |
resolve collisions and contested cells
        |
commit surviving movements
        |
update habits and population state
```

Make clear which differences are architectural and which change the model's substantive timing assumptions.

== Forecast-and-control extension

Use the existing occupancy-strategy analysis as evidence that separating prediction from choice makes behavioral mechanisms independently substitutable. Focus the main text on:

- Naive current-lane prediction.
- Two-frame temporal-consistency prediction.
- Iterated decision-aware prediction.

The remaining strategies can be a compact robustness table or appendix. Emphasize that a predicted state affects decisions and therefore changes the state being predicted: these are forecast-and-control policies, not passive forecasts.

= Comparative Evaluation

== Evaluation logic

The paper cannot prove an architectural advantage from one attractive code example. Combine a design comparison, behavioral experiments, and performance measurement.

#table(
  columns: (1.1fr, 1.6fr, 1.7fr),
  inset: 6pt,
  align: (left, left, left),
  table.header([*Claim*], [*Evidence*], [*Threat to address*]),
  [Compositional heterogeneity], [Add and remove capabilities without defining combination-specific types], [Idiomatic OOP can also use composition],
  [System-centered modularity], [Substitute mechanisms while sharing state and unrelated processes], [Lines of code alone are a weak measure],
  [Natural simultaneous interaction], [Derive staged formulation from system dependencies], [Other ABM frameworks can also use buffers and phases],
  [Parallel scalability], [Thread, memory, and possibly GPU benchmarks], [Interaction conflicts may dominate],
)

== Experiment A: Representing heterogeneity

- Define a set of independent behavioral capabilities.
- Construct populations with overlapping component signatures.
- Add one new capability after both implementations are complete.
- Compare the required changes to state definitions, behavior dispatch, initialization, logging, and analysis.
- Report qualitative dependency changes and limited quantitative measures such as touched modules, duplicated transition logic, and combinations represented.

The conclusion should concern locality and composability, not the impossibility of implementing the same model with objects.

== Experiment B: Substituting systems

- Hold entities and unrelated processes constant.
- Exchange prediction, learning, or conflict-resolution systems.
- Verify that unaffected mechanisms are reused without conditional branches.
- Use the existing strategy results to illustrate scientific experiments enabled by this separation.

== Experiment C: Interaction semantics

Compare fixed sequential, shuffled sequential, synchronous, and staged interaction only where they are scientifically meaningful. Hold behavioral equations and initial conditions constant. Measure:

- Convention strength and persistence.
- Time to convention.
- Collision events and removed agents.
- Throughput or successful movements.
- Sensitivity to density and capability composition.

This experiment supports the interaction argument; it should not displace the broader architectural contribution.

== Experiment D: Engineering performance

Benchmark at least:

+ Agent-centered serial implementation.
+ ECS serial implementation.
+ ECS threaded implementation.
+ ECS GPU implementation, only if it is complete enough for a fair comparison.

Report:

- Wall-clock time after compilation and warm-up.
- Population and component-count scaling.
- Allocations and peak memory.
- Cache misses and memory bandwidth where measurable.
- Thread scaling and parallel efficiency.
- Time spent in queries, systems, structural changes, synchronization, and conflict resolution.
- Hardware, software versions, compiler settings, and number of repetitions.

Use both compute-light and interaction-heavy workloads. A synthetic component-sweep benchmark can isolate memory layout, but the traffic model must show end-to-end behavior.

== Randomness, inference, and reproducibility

- Pre-generate or index stochastic draws by replicate, time, entity, and event so architectural treatments do not merely consume different RNG streams.
- Distinguish matched initialization from genuinely paired subsequent shocks.
- Predefine primary contrasts and outcome measures.
- Report Monte Carlo uncertainty and effect sizes.
- Use longer horizons and multiple initial conditions.
- Archive code, project manifests, raw replicate data, and exact reproduction commands.

= Results

#scaffold-note([Do not turn this section into prose yet], [
  Organize results by research question, not by the order in which scripts were written. Each subsection should begin with a one-sentence answer, then show the evidence and its uncertainty.
])

== RQ1: Compositional heterogeneity

- Report the number and distribution of component signatures used in the experiment.
- Show how a capability is introduced or removed in both architectures.
- Report code-locality or dependency evidence without treating code size as scientific proof.
- Analyze whether structural heterogeneity changes convention formation beyond parametric heterogeneity.

== RQ2: System-centered model organization

- Present the system dependency graph.
- Show which systems are reused across behavioral variants.
- Report unit and integration tests at system boundaries.
- Discuss whether the executable structure corresponds more directly to the published process description.

== RQ3: Simultaneous and parallel interactions

- Compare sequential and staged semantics.
- Quantify update-order sensitivity.
- Show whether thread count changes results under a fixed declared semantics.
- Report conflict frequency and resolution cost.

== RQ4: Engineering performance

- Separate data-layout gains from threading gains.
- Show scaling curves rather than a single population size.
- Identify cross-over points where ECS overhead becomes worthwhile.
- Report workloads where ECS provides little or no advantage.

== Supporting forecast-strategy results

Current exploratory findings to preserve and later verify:

- Decision-aware prediction produces the strongest survival and replacement outcomes among the tested policies.
- Two-frame Naive provides a simpler improvement using only current-time information.
- The ranking is robust across the tested densities and much of the behavioral-weight grid.
- These findings demonstrate mechanism substitution and coupled forecast-and-control, not by themselves the superiority of ECS.

= Discussion

== The agent without an agent object

Address whether ECS undermines autonomy. Proposed position:

- An agent is a scientific unit with identity, state, information, and possible actions.
- A software object is only one implementation of that unit.
- ECS externalizes shared transition laws while retaining entity-specific state.
- Agency can therefore remain meaningful without behavior being stored in an object method.

== What ECS changes for modelers

- Model construction begins with processes and required state rather than a complete taxonomy of actors.
- Heterogeneity can be expressed through overlapping capabilities.
- Model mechanisms can be tested and substituted as systems.
- Timing, conflicts, and dependencies must be made explicit.
- The same explicitness can support deterministic concurrency and efficient batch execution.

== What ECS does not solve

- It does not determine the scientifically correct timing semantics.
- It does not make conflicting interactions automatically parallel.
- It does not guarantee cache or GPU performance.
- It does not remove the need for validation, calibration, or sensitivity analysis.
- It may make an individual agent's complete behavior harder to inspect.
- It can be excessive for small models with a single stable agent type and strongly individualized logic.

== Implications for economic ABMs

Develop examples beyond traffic:

- A firm can acquire exporter, borrower, employer, or innovator capabilities without becoming a new nominal type for every combination.
- A household can participate in labor, credit, housing, and consumption systems according to its current components.
- Institutions can be modeled as systems or resources rather than necessarily as agents, forcing the modeler to state where agency is substantively intended.
- Entry, bankruptcy, learning, and institutional change can be represented as transformations of component composition.

These examples should remain implications unless they are implemented as additional case studies.

== Threats to validity

- One traffic model cannot establish universal architectural superiority.
- The agent-centered implementation may reflect author familiarity or framework-specific constraints.
- The chosen ECS library may conflate the abstract pattern with a particular storage implementation.
- Performance results may be hardware- and workload-specific.
- Dynamic composition can introduce semantic choices about when component changes take effect.
- The distinction between an entity, an agent, and an environmental object must be documented explicitly.

= Conclusion

Restate the intended conclusion cautiously:

Entity Component Systems should be evaluated in ABM not only as a performance technique but as an alternative architecture for model specification. Their main scientific promise lies in representing heterogeneous agents as changing compositions of capabilities, organizing behavior as explicit population-level processes, and exposing the dependencies involved in simultaneous interactions. Cache-efficient storage and parallel execution are important consequences, but their benefits remain empirical and workload-dependent.

The final paragraph should identify the next research step: application to a larger economic model with multiple institutional roles and endogenous changes in agent capabilities.

= Appendix Roadmap

== Complete ODD description

- Purpose and patterns.
- Entities, state variables, and scales.
- Process overview and scheduling.
- Design concepts.
- Initialization.
- Input data.
- Submodels and equations.

== Architecture comparison

- Full agent-centered pseudocode.
- Full ECS system table with queries, reads, writes, and structural changes.
- Dependency graph and execution phases.
- Definition of every architecture-specific deviation.

== Parameters and experimental design

- Default parameters and scientific interpretation.
- Capability-composition treatments.
- Density, horizon, and sensitivity grids.
- Replicate counts and seed construction.
- Primary and secondary outcomes.

== Additional results

- Full forecast-strategy tables.
- Weight sensitivity.
- Longer-horizon and initialization robustness.
- Convergence diagnostics for iterated decision-aware prediction.
- Complete performance profiles.

== Reproducibility

- Repository and archived release.
- Julia, Rust, Typst, and package versions.
- Hardware and operating-system description.
- Commands for tests, simulations, figures, benchmarks, and paper compilation.
