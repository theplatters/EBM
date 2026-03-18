#set document(title: "Rethinking ABM Architecture: Entity Component Systems and the Case for Parallel Agent Interactions")
#set page(margin: (x: 2.5cm, y: 2.5cm))
#set text(font: "New Computer Modern", size: 11pt)
#set par(justify: true, leading: 0.65em)

// --- Title ---
#align(center)[
  #text(size: 16pt, weight: "bold")[
    Rethinking ABM Architecture: Entity Component Systems \ and the Case for Parallel Agent Interactions
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

  #text(size: 10pt)[March 2026]

  #v(1.2em)
]

// --- Abstract ---
#heading(level: 1, numbering: none)[Abstract]

Agent-based modeling (ABM) has cemented itself as one of the core approaches in heterodox economics, capturing complex system dynamics through the interaction of individual agents governed by manageable rulesets. Its appeal lies in its ability to generate emergent macro-level phenomena from micro-level behavioral rules without requiring representative agents or equilibrium assumptions. However, despite its theoretical flexibility, most ABM frameworks rely on sequential agent updating, where one iterates through agents one at a time in a fixed or shuffled order. This introduces update-order artifacts: the sequence in which agents act can materially alter simulation outcomes, creating a methodological confound that is rarely acknowledged and even more rarely addressed. Beyond this epistemological concern, sequential processing imposes practical limits on computational scalability, constraining the size and complexity of the populations that can be feasibly modeled.

We propose adopting Entity Component Systems (ECS) as an alternative architectural foundation for ABM. Originating in high-performance game development, ECS decouples agents (entities) from their properties (components) and behavioral rules (systems). This separation of concerns yields a data-oriented memory layout that enables efficient parallel processing of homogeneous operations across large agent populations. Where traditional object-oriented ABM frameworks bind data and behavior tightly within each agent, ECS organizes data by type rather than by owner, allowing a single system to operate on thousands or millions of components simultaneously.

We explore the practical implications of this architectural choice through hands-on examples, revisiting and recreating the classical ABM presented by #cite(<hodgsonEconomicsShadowsDarwin2006>, form: "prose"), which models generalized Darwinian selection processes in an economic context. We first replicate their original sequential model within an ECS framework, verifying behavioral equivalence, and then extend it by adapting the behavioral rules to a fully parallel approach in which all agents observe and act on the same world state simultaneously. This parallel formulation eliminates update-order dependence and offers a more faithful representation of environments where agents plausibly act concurrently rather than taking turns.

Comparing the sequential and parallel implementations, we examine how the choice of execution paradigm affects emergent dynamics, equilibrium selection, and the robustness of results. More broadly, we argue that programming paradigms and technical limitations are not neutral scaffolding, they quietly shape our core assumptions about what agents can perceive, when they can act, and how they interact. By making these architectural choices explicit, ECS offers heterodox economists not only a more scalable simulation tool but also a sharper lens through which to interrogate the hidden assumptions embedded in their models.


#bibliography("Econ.bib")
