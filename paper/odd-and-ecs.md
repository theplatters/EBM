# ODD and ECS: how the description protocol fits this paper (working note)

## Why ODD matters for this manuscript

ODD (Overview, Design concepts, Details) is the community standard for
describing agent-based models (Grimm et al. 2006, 2010; second update
2020, already in `Econ.bib` as `grimmODDProtocolDescribing2020`). The
scaffold note in the literature chapter explicitly asks for ODD
literature, and the paper already contains a "Complete ODD description"
appendix (`sec:odd-appendix`, fed by `sugarscape_model_specification`).

The connection to the thesis is direct: the paper argues ECS is an
*architecture for scientific model specification*, not merely a storage
optimization. ODD defines what a scientific model specification should
contain and communicate. ECS can then be evaluated by how well its
declarations cover, and even generate, the parts of ODD that are
notoriously underspecified in prose — above all process overview and
scheduling, which is exactly the paper's RQ3 dimension.

## Element-by-element mapping

| ODD element | Paper concept | ECS artifact in the implementation |
| --- | --- | --- |
| Purpose and patterns | Purpose block + primary outcomes (Gini, wealth, prevalence) | Logger schema; experiment outcome lists |
| Entities, state variables, scales | Agent descriptor `a_W(i) = (tau, kappa, X, x)`; grid/population/horizon | Entities, components, resources; component signatures |
| Process overview and scheduling | Schedule semantics: phase order, visibility, update timing, equivalence conditions | Declared system order per period; read/write sets; phase boundaries |
| Design concepts (emergence, sensing, interaction, stochasticity, observation) | Behavior organization; simultaneous interaction semantics | Queries; multi-query systems with match relations; seeded RNG as a resource; `logger!` system |
| Initialization | State and initialization section | Setup systems: landscape seeding, citizen placement |
| Input data | Parameters | Resources (`ModelParams`), capacity matrices |
| Submodels | Individual mechanisms | One system per mechanism: query family `Q_k`, Read/Write sets, `F_k` |

The current appendix text already follows this structure informally
(purpose → state/initialization → process overview → movement →
disease/lifecycle → recorded outcomes). It only needs ODD element names
made explicit to be a compliant ODD description.

## Where the alignment is strongest

1. **Scheduling.** ODD requires that activation order, simultaneity, and
   update visibility be stated. The paper's contribution is showing that
   agent-object code makes these properties hard to state *and hard to
   verify from the implementation*, while ECS declares them (system
   order, phase boundaries, read/write sets, RNG partitioning). The
   disease-before-lifecycle example (`sec:breakdown`) is an ODD
   scheduling constraint that the idiom cannot honor — usable as a
   concrete illustration of why ODD element 3 needs machine-level
   support.
2. **Submodels with contracts.** The planned supplementary "full ECS
   system table with queries, reads, writes, and structural changes" is
   essentially a machine-checkable rendering of ODD elements 3 and 7.
   Each system specification `(Q_k, Read_k, Write_k, F_k)` *is* a
   submodel contract plus its scheduling inputs.
3. **Traceability.** A known ODD weakness is drift between the written
   protocol and the code. Because queries and access declarations are
   metadata, large parts of the ODD process-overview and submodel
   sections could be generated from `src/Sugarscape/` and diffed against
   the hand-written appendix. The mutation battery plays the same role
   at runtime: it detects where implementation and specification
   diverge.

## Where the alignment breaks down (keep these qualifications)

- ODD carries purpose, design-concept rationale, and empirical patterns.
  No code artifact, ECS or otherwise, can generate those; they are
  scientific judgments. Read/write sets are conservative summaries, not
  scientific justification (the paper already states this).
- ODD is deliberately implementation- and framework-neutral. The ODD
  appendix should stay in scientific vocabulary (entities, state,
  processes, periods); the ECS mapping belongs in the supplementary
  architecture comparison, not inside the ODD text itself.
- Design concepts map only partially: "learning" and "prediction" have
  no distinctive ECS counterpart in this model; "adaptation" is just a
  system acting on trait components.
- ODD describes the model once; ECS signatures describe state *at a
  world state* (`kappa_W`). Dynamic compositional heterogeneity means
  the "Entities, state variables" section must describe the admissible
  component set, not a fixed schema — worth one sentence in the appendix.

## Concrete suggestions for the manuscript

1. Cite `grimmODDProtocolDescribing2020` in `sec:literature` (the
   scaffold note already requests this) and, if added to the bib,
   Grimm et al. 2006/2010 for the original protocol.
2. Re-label the appendix headings to the ODD element names (Purpose and
   patterns; Entities, state variables, and scales; Process overview and
   scheduling; Design concepts; Initialization; Input data; Submodels)
   so `sec:odd-appendix` is explicitly a compliant ODD.
3. Add a short passage (ECS chapter or appendix intro) stating which ODD
   elements are machine-derivable from the declarations (entities,
   process overview, initialization, submodels) and which are not
   (purpose, design-concept rationale).
4. In the supplementary architecture comparison, present the system
   table as the executable counterpart of ODD's process-overview and
   submodels sections — this strengthens the "specification, not just
   optimization" positioning without new claims.
