# ECS and Agents.jl Sugarscape comparison

This comparison covers three outcome-equivalent implementations of the repository's
single-resource Sugarscape: the Ark.jl ECS implementation with either movement mode,
`AgentSequential`, and `AgentSynchronous`. The equivalence suite compares every citizen
field exposed by `CitizenState`, the complete landscape matrix, and every logger series
at initialization and after every step. It also forces synchronous contention,
old-age replacement, starvation without replacement, disease transmission and
recovery, and reproduction. The results below therefore compare implementation
architectures executing the same model semantics, not merely similar variants.

## Performance

The benchmark measures a complete setup plus 100 model steps. Each row is the median
of 20 independently initialized runs after one unmeasured compilation/warm-up run.
Each ECS/Agents.jl pair uses the same parameters and seed. Measurements were made on
2026-08-14 with Julia 1.12.6, BenchmarkTools 1.7.0, one Julia thread, and an AMD Ryzen
7 5825U CPU on Linux.

The baseline uses a 50 × 50 grid, 400 initial citizens, seed 20260813, and the default
replacement regime. The extended case uses the same size, population, seed, and
horizon, with replacement disabled, reproduction enabled at probability 0.02, and an
original-style catalogue of 10 diseases with four initially assigned to every citizen;
all other parameters retain their defaults.

| Scenario | Implementation | Median time (ms) | Median memory (MiB) | Allocations |
|---|---|---:|---:|---:|
| Baseline | ECS sequential | 12.819 | 2.933 | 12,091 |
| Baseline | Agents.jl sequential | 17.104 | 2.440 | 8,329 |
| Baseline | ECS synchronous | 15.175 | 4.574 | 16,009 |
| Baseline | Agents.jl synchronous | 21.426 | 4.334 | 15,133 |
| Extended | ECS sequential | 12.571 | 12.421 | 44,253 |
| Extended | Agents.jl sequential | 12.025 | 2.111 | 8,415 |
| Extended | ECS synchronous | 11.509 | 11.371 | 39,226 |
| Extended | Agents.jl synchronous | 12.430 | 3.195 | 14,307 |

ECS is 25.1% faster than Agents.jl sequential and 29.2% faster than Agents.jl
synchronous in the baseline. In the extended case the runtimes are closer: Agents.jl
sequential is 4.3% faster, while ECS synchronous is 7.4% faster. The feature-heavy ECS
runs allocate more memory because infection and reproduction require staged structural
changes. Comparisons should be made within a scenario and movement mode, not between
scenarios.

These are measurements of the checked-in code on one machine, not universal framework
performance claims. Runtime differences of this size can change with Julia, package,
hardware, garbage-collector, and model-parameter changes. Reproduce the table with:

```sh
julia --project=. Sugarscape/agent_oriented/benchmark.jl
```

The script accepts `SUGARSCAPE_BENCH_SEED`, `SUGARSCAPE_BENCH_STEPS`,
`SUGARSCAPE_BENCH_WIDTH`, `SUGARSCAPE_BENCH_HEIGHT`,
`SUGARSCAPE_BENCH_POPULATION`, `SUGARSCAPE_BENCH_SAMPLES`, and
`SUGARSCAPE_BENCH_THREADED` overrides.

### Multithreaded scaling

Threading is deliberately workload-gated. Small/default simulations use the serial
kernels, avoiding scheduler overhead. Large synchronous simulations score movement
destinations in parallel, then perform seeded tie-breaking and state mutation serially
in stable citizen-ID order. Shuffled-sequential movement is order-dependent and remains
serial. Its candidate parallel growback and setup kernels were rejected because they did
not substantially improve complete setup-and-run benchmarks.

Independent paired benchmarks on the same Ryzen 7 5825U showed:

| Implementation | Threads | Scaling workload | Serial (ms) | Threaded (ms) | Speedup |
|---|---:|---|---:|---:|---:|
| ECS synchronous | 4 | 300×300, 30,000 citizens, vision 40–50, 10 steps | 1167.72 | 375.71 | 3.11× |
| ECS synchronous | 8 | same | 1446.19 | 624.52 | 2.32× |
| Agents.jl synchronous | 4 | 200×200, 10,000 citizens, vision 8–12, 20 steps | 419.40 | 318.78 | 1.32× |
| Agents.jl synchronous | 8 | same | 267.06 | 176.55 | 1.51× |

Each timing covers complete setup and execution and reports the median of warmed
serial/threaded runs. The 50×50, 400-citizen default remains below the threading
thresholds; repeated comparisons found no material runtime or allocation regression.
The synchronous implementations require at least four Julia threads, 2,000 citizens,
and 20,000 estimated visibility checks before parallel planning.

Run serial and threaded comparisons in processes with the same thread count by setting
`SUGARSCAPE_BENCH_THREADED=false` and `true`, respectively. For example:

```sh
SUGARSCAPE_BENCH_THREADED=false julia -t 8 --project=. \
  Sugarscape/agent_oriented/benchmark.jl
SUGARSCAPE_BENCH_THREADED=true julia -t 8 --project=. \
  Sugarscape/agent_oriented/benchmark.jl
```

## Lines of code

The count below is source lines of code (SLOC): nonblank lines whose first
non-whitespace character is not `#`. It includes imports, declarations, docstrings,
and module boilerplate, and excludes tests, plotting, interactive visualization,
documentation, and this benchmark. This deliberately simple definition is reproducible
without a language-specific counter:

```sh
awk 'NF && $1 !~ /^#/ {n++} END {print n}' \
  Sugarscape/components/*.jl Sugarscape/core/*.jl Sugarscape/systems/*.jl \
  Sugarscape/simulation/*.jl Sugarscape/analysis/logger.jl \
  Sugarscape/analysis/runner.jl
awk 'NF && $1 !~ /^#/ {n++} END {print n}' \
  Sugarscape/agent_oriented/sequential.jl
awk 'NF && $1 !~ /^#/ {n++} END {print n}' \
  Sugarscape/agent_oriented/synchronous.jl
```

| Implementation scope | SLOC | Interpretation |
|---|---:|---|
| ECS computational model, both movement modes | 1,486 | Shared components, resources, features, logging, setup, and runner |
| Agents.jl sequential | 728 | Self-contained sequential model |
| Agents.jl synchronous | 979 | Self-contained synchronous model |
| Both Agents.jl files | 1,707 | Includes duplicated setup, disease, lifecycle, reproduction, logging, and buffers |

The fairest conclusion depends on the unit of comparison. One Agents.jl variant remains
substantially smaller than the dual-mode ECS implementation. Providing both standalone
Agents.jl variants, however, takes 221 more SLOC than the shared dual-mode ECS model.
The reusable-buffer implementations add explicit scratch-state declarations and reset
logic to each architecture. Dividing all ECS lines between its modes would double-count
shared infrastructure, while comparing the full ECS scope against only one Agents.jl
file would charge ECS for its second movement mode. The table reports both views instead
of assigning shared lines arbitrarily.

## Architecture

### State representation

The ECS citizen is an entity assembled from narrow components. Identity, position,
proposal, vision, metabolism, sugar, age, maximum age, and initial endowment are
separate value types. Sex is represented by mutually exclusive `Female`/`Male` tag
components. Infection is structural: an infected citizen has an `Infection` component,
whose bit mask identifies every disease carried from the shared catalogue, and recovery
removes it once the mask is empty. Persistent world state lives in resources: parameters,
the disease catalogue, seeded RNG, clock, next citizen ID, landscape, occupancy grid,
step events, and logger.

Each Agents.jl citizen is instead one mutable `@agent` record containing spatial state,
traits, wealth, age, sex as a `Symbol`, immune genotype and trained phenotype, and
`infection::Union{Nothing, Infection}`. Infection changes a field value rather than the
agent's structure. Parameters, landscape, event counters, clock, and logger are model
properties. `AgentSequential` delegates occupancy to `GridSpace`; `AgentSynchronous`
also maintains an explicit occupancy matrix because all proposals must observe the same
pre-movement state.

This makes per-citizen state locally readable in the agent-oriented versions: one agent
object shows nearly everything about that citizen. The ECS representation makes data
roles and structural heterogeneity explicit and lets systems query only the components
they require. Its cost is more wrapper types, resources, queries, and conversion between
component records and analysis snapshots.

### Behavior and schedule

ECS behavior is split by concern across `systems/growback.jl`, `movement.jl`,
`disease.jl`, `lifecycle.jl`, and `reproduction.jl`. `simulation/step.jl` is the explicit
orchestrator. Systems query component columns, stage structural changes, and commit
entity additions/removals outside active queries.

The Agents.jl versions organize the same phases as functions inside each movement-mode
module. Their step functions retain the same order:

```text
growback -> movement -> disease -> lifecycle -> reproduction -> clock -> logger
```

This preserves outcome equivalence, but feature code is duplicated between the two
standalone modules. The duplication makes each model independently readable and
modifiable; the ECS version centralizes shared features and selects only the movement
system within one schedule.

### Sequential and synchronous coordination

For sequential movement, both versions sort citizens by stable ID, shuffle that list
with the seeded model RNG, and immediately expose each move and harvest to later
citizens. ECS updates an `OccupancyGrid` resource and later applies component values;
Agents.jl uses `move_agent!` and the grid's position index.

For synchronous movement, both versions first select proposals from the same occupancy
and landscape, group contenders by destination, sort destinations and IDs, use the
seeded RNG to select conflict winners, and then commit moves. ECS stores proposals in a
`ProposedPosition` component. `AgentSynchronous` stores `proposed_pos` in every mutable
agent and uses its explicit occupancy matrix. This is the main reason the synchronous
Agents.jl model is longer than the sequential one: synchronous semantics require an
agent-oriented implementation to add an explicit proposal/commit protocol rather than
relying on ordinary one-agent-at-a-time activation.

### Practical trade-off

The agent-oriented representation is compact for a single movement regime and keeps
all individual state together. The ECS representation is more decomposed and has a
larger fixed vocabulary, but shares features cleanly across movement regimes and makes
optional structure and system data dependencies visible. In this implementation,
architecture choice is more consequential for organization and extensibility than for
speed: warmed end-to-end timings are close and reverse ordering across the two tested
scenarios.
