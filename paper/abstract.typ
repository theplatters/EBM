#set document(title: "Beyond the Agent Object: Entity Component Systems as an Architecture for Agent-Based Modeling")
#set page(margin: (x: 2.5cm, y: 2.5cm))
#set text(font: "New Computer Modern", size: 11pt)
#set par(justify: true, leading: 0.65em)

// --- Title ---
#align(center)[
  #text(size: 16pt, weight: "bold")[
    Beyond the Agent Object: Entity Component Systems \ as an Architecture for Agent-Based Modeling
  ]

  #v(0.8em)

  // --- Author ---
  #text(size: 12pt)[Franz Scharnreitner]
  #linebreak()
  #text(size: 10pt, style: "italic")[
    Institute for the comprehensive analysis of the economy, Johannes Kepler Universität Linz\
    #link("mailto:franz.scharnreitner@jku.at")
  ]

  #v(0.4em)

  #text(size: 10pt)[August 2026]

  #v(1.2em)
]

// --- Abstract ---
#heading(level: 1, numbering: none)[Abstract]

Agent-based models are commonly implemented around agent objects that combine
identity, state, and behavior. Although this organization mirrors the intuitive
description of autonomous agents, it makes the complete agent representation
the principal unit of model construction and can obscure the population-level
processes through which agents interact. We examine Entity Component Systems
(ECS) as an alternative architecture in which entities supply identity,
components represent state and roles, and systems transform populations selected
by component queries.

The comparison addresses three questions. First, how does ECS affect the
representation and modification of overlapping and changing roles relative to
an idiomatic agent-centered implementation of the same model? Second, how does
system organization affect the locality, reuse, substitutability, and
inspectability of population mechanisms? Third, how do dependencies and phase
boundaries define information visibility and update timing, and under what
conditions do alternative execution orders preserve outcomes?

The comparative case is the single-resource Sugarscape wealth-distribution
model introduced by #cite(<epsteinGrowingArtificialSocieties1996>, form:
"prose"). Heterogeneous citizens move and harvest on a regenerating landscape,
metabolize sugar, accumulate wealth, age, and die. Optional reproduction and
disease add overlapping and changing roles: sex is represented by exclusive
tags, infection by a component added on transmission and removed on recovery,
and fertility by predicates over age, wealth, sex, and neighborhood state.
Movement is implemented under both shuffled-sequential and staged synchronous
semantics, making the consequences of observation and commitment timing
explicit.

The evaluation pairs the ECS model with a semantics-matched, idiomatic
agent-centered reference and separates architecture treatments from schedule
treatments. Storage layout, cache behavior, and multicore scaling are assessed
as secondary engineering consequences rather than as a separate research
question. The paper thereby treats ECS not as an automatic route to parallelism
or superior performance, but as an architecture that can make state
composition, population mechanisms, and schedule commitments explicit and
testable.


#bibliography("Econ.bib")
