# Paired risk-aversion experiment.

# The three `fixed_*` treatments are interventions, not replacement policies:
# the normal EntryDrawReplacement genome is still made by the core, then every
# RiskAversion component is overwritten before the next speed decision. This
# uses no random draws. The two uniform treatments retain the core's uniform
# entry (or evolutionary-parent) semantics.
using CairoMakie
using EBM
using Statistics

include(joinpath(@__DIR__, "social_habit_common.jl"))
using .SocialHabitExperiments

const T = EBM.Traffic
const TREATMENTS = (
    (name = "fixed_0.0", policy = "fixed risk aversion", replacement = :entry, fixed = 0.0),
    (name = "fixed_0.5", policy = "fixed risk aversion", replacement = :entry, fixed = 0.5),
    (name = "fixed_1.0", policy = "fixed risk aversion", replacement = :entry, fixed = 1.0),
    (name = "uniform_entry", policy = "uniform entry risk", replacement = :entry, fixed = NaN),
    (name = "uniform_evolutionary", policy = "uniform evolutionary risk", replacement = :evolutionary, fixed = NaN),
)
const RESULT_COLUMNS = (
    :seed, :treatment, :policy, :fixed_risk_aversion, :steps, :burn_in, :population,
    :ring_x, :ring_y, :lookahead, :error_rate, :mixture_share,
    :capability_mutation_rate, :trait_mutation_scale, :mean_convention_strength,
    :coordinated_fraction, :time_to_convention, :completed_cells_per_car_step,
    :mean_proposed_speed, :mean_realized_speed, :dangerous_proposal_acceptance,
    :replacement_pressure, :population_risk_mean, :population_risk_std,
    :survivor_risk_mean, :selected_parent_risk_mean, :final_risk_mean,
    :final_risk_std, :final_risk_q10, :final_risk_median, :final_risk_q90,
)
const DISTRIBUTION_COLUMNS = (:seed, :treatment, :phase, :car_index, :risk_aversion)

env_int(name, default) = parse(Int, get(ENV, name, string(default)))
env_float(name, default) = parse(Float64, get(ENV, name, string(default)))
replicates = env_int("TRAFFIC_REPLICATES", 30)
first_seed = env_int("TRAFFIC_FIRST_SEED", 20260901)
steps = env_int("TRAFFIC_STEPS", 5_000)
burn_in = env_int("TRAFFIC_BURN_IN", 1_000)
config = ExperimentConfig(
    seeds = first_seed:(first_seed + replicates - 1), steps = steps, burn_in = burn_in,
    population = env_int("TRAFFIC_POPULATION", 120), ring_y = env_int("TRAFFIC_RING_Y", 300),
    lookahead = env_int("TRAFFIC_LOOKAHEAD", 20),
    capability_mutation_rate = env_float("TRAFFIC_MUTATION_RATE", 0.02),
    trait_mutation_scale = env_float("TRAFFIC_MUTATION_SCALE", 0.05),
)
SocialHabitExperiments.validate(config)

function risk_values(world)
    values = Tuple{Tuple{Int,Int},Float64}[]
    for (entities, risks) in T.Query(world, (T.RiskAversion,))
        for i in eachindex(entities)
            entity = entities[i]
            push!(values, ((getfield(entity, :_id), getfield(entity, :_gen)), risks[i].value))
        end
    end
    sort!(values; by = first)
    return last.(values)
end

function set_fixed_risk!(world, value)
    isnan(value) && return nothing
    for (_, risks) in T.Query(world, (T.RiskAversion,))
        for i in eachindex(risks)
            risks[i] = T.RiskAversion(value)
        end
    end
    return nothing
end

function treatment_model(config, treatment)
    scenario = treatment.replacement == :evolutionary ?
        :mixture_evolutionary_replacement : :mixture_static_replacement
    return capability_model(config, scenario)
end

function run_condition(config, seed, treatment)
    model = treatment_model(config, treatment)
    world = T.setup_world(T.ModelArgs(seed = seed, params = SocialHabitExperiments.model_params(config),
                                      prediction_strategy = model, steps = 0))
    # Capture in entity order, making distribution rows reproducible.
    set_fixed_risk!(world, treatment.fixed)
    initial = risk_values(world)
    conventions = Float64[]
    proposed_total = proposed_count = realized_total = realized_count = 0
    dangerous = accepted = deaths = 0
    population_risks = Float64[]
    survivor_risks = Float64[]
    parent_risks = Float64[]
    time_to_convention = nothing
    for step in 1:config.steps
        # This also covers newborns from the preceding tick.  It deliberately
        # does not alter EntryDrawReplacement's RNG consumption or genomes.
        set_fixed_risk!(world, treatment.fixed)
        T.step!(world, model)
        strength = SocialHabitExperiments.convention_strength(world)
        isnothing(time_to_convention) && strength >= config.convention_target &&
            (time_to_convention = step)
        if step > config.burn_in
            diagnostics = T.Ark.get_resource(world, T.CapabilityTickDiagnostics)
            logger = T.Ark.get_resource(world, T.Logger)
            # Apply the intervention before any post-step population sample.
            # This changes components only and deliberately consumes no RNG.
            set_fixed_risk!(world, treatment.fixed)
            push!(conventions, strength)
            proposed_total += diagnostics.proposed_speed_total
            proposed_count += diagnostics.proposed_speed_count
            realized_total += diagnostics.realized_speed_total
            realized_count += diagnostics.realized_speed_count
            dangerous += diagnostics.dangerous_proposals
            accepted += diagnostics.accepted_dangerous_proposals
            deaths += isempty(logger.deaths) ? 0 : last(logger.deaths)
            append!(population_risks, risk_values(world))
            append!(survivor_risks, diagnostics.survivor_risks)
            append!(parent_risks, diagnostics.selected_parent_risks)
        end
    end
    set_fixed_risk!(world, treatment.fixed)
    final = risk_values(world)
    car_steps = (config.steps - config.burn_in) * config.population
    average(values) = isempty(values) ? NaN : mean(values)
    return (
        seed = seed, treatment = treatment.name, policy = treatment.policy,
        fixed_risk_aversion = treatment.fixed, steps = config.steps, burn_in = config.burn_in,
        population = config.population, ring_x = 2, ring_y = config.ring_y,
        lookahead = config.lookahead, error_rate = config.error_rate,
        mixture_share = config.mixture_share,
        capability_mutation_rate = config.capability_mutation_rate,
        trait_mutation_scale = config.trait_mutation_scale,
        mean_convention_strength = average(conventions),
        coordinated_fraction = average(conventions .>= config.convention_target),
        time_to_convention = something(time_to_convention, NaN),
        completed_cells_per_car_step = realized_total / car_steps,
        mean_proposed_speed = proposed_count == 0 ? NaN : proposed_total / proposed_count,
        mean_realized_speed = realized_count == 0 ? NaN : realized_total / realized_count,
        dangerous_proposal_acceptance = dangerous == 0 ? NaN : accepted / dangerous,
        replacement_pressure = deaths / car_steps,
        population_risk_mean = average(population_risks),
        population_risk_std = length(population_risks) < 2 ? 0.0 : std(population_risks),
        survivor_risk_mean = average(survivor_risks),
        selected_parent_risk_mean = average(parent_risks),
        final_risk_mean = average(final), final_risk_std = length(final) < 2 ? 0.0 : std(final),
        final_risk_q10 = isempty(final) ? NaN : quantile(final, 0.10),
        final_risk_median = isempty(final) ? NaN : quantile(final, 0.50),
        final_risk_q90 = isempty(final) ? NaN : quantile(final, 0.90),
        initial = initial, final = final,
    )
end

jobs = [(seed, treatment) for seed in config.seeds for treatment in TREATMENTS]
rows = Vector{NamedTuple}(undef, length(jobs))
completed = Threads.Atomic{Int}(0)
progress_lock = ReentrantLock()
Threads.@threads for i in eachindex(jobs)
    seed, treatment = jobs[i]
    rows[i] = run_condition(config, seed, treatment)
    count = Threads.atomic_add!(completed, 1) + 1
    lock(progress_lock) do
        println("completed $count/$(length(jobs)): seed=$seed, treatment=$(treatment.name)")
    end
end

output_dir = get(ENV, "TRAFFIC_OUTPUT_DIR", joinpath(@__DIR__, "..", "plots"))
mkpath(output_dir)
function write_csv(path, columns, rows)
    open(path, "w") do io
        println(io, join(string.(columns), ','))
        for row in rows
            println(io, join((string(getproperty(row, column)) for column in columns), ','))
        end
    end
    return path
end
run_path = write_csv(joinpath(output_dir, "risk_aversion_runs.csv"), RESULT_COLUMNS, rows)
distribution_rows = NamedTuple[]
for row in rows
    for (phase, values) in (("initial", row.initial), ("final", row.final))
        for (index, value) in enumerate(values)
            push!(distribution_rows, (seed = row.seed, treatment = row.treatment,
                                      phase = phase, car_index = index, risk_aversion = value))
        end
    end
end
distribution_path = write_csv(joinpath(output_dir, "risk_aversion_distributions.csv"),
                              DISTRIBUTION_COLUMNS, distribution_rows)

colors = Dict(t.name => c for (t, c) in zip(TREATMENTS, (:gray40, :steelblue, :firebrick2, :darkorange2, :seagreen3)))
display_names = Dict(
    "fixed_0.0" => "fixed risk 0.0",
    "fixed_0.5" => "fixed risk 0.5",
    "fixed_1.0" => "fixed risk 1.0",
    "uniform_entry" => "uniform entry",
    "uniform_evolutionary" => "uniform evolutionary",
)
metrics = ((:mean_convention_strength, "convention strength"),
           (:coordinated_fraction, "coordinated fraction"),
           (:completed_cells_per_car_step, "completed cells per car-step"),
           (:mean_proposed_speed, "mean proposed speed"),
           (:dangerous_proposal_acceptance, "perceived-danger acceptance"),
           (:replacement_pressure, "collision replacements per car-step"))
figure = Figure(size = (1500, 1050), fontsize = 15)
for (panel, (metric, label)) in enumerate(metrics)
    axis = Axis(figure[(panel - 1) ÷ 2 + 1, (panel - 1) % 2 + 1], ylabel = label,
                xlabel = panel > 4 ? "treatment" : "")
    for (index, treatment) in enumerate(TREATMENTS)
        values = [getproperty(row, metric) for row in rows if row.treatment == treatment.name]
        values = filter(!isnan, values)
        scatter!(axis, fill(index, length(values)), values; color = (colors[treatment.name], .35))
        !isempty(values) && (scatter!(axis, [index], [mean(values)]; color = colors[treatment.name], markersize = 13);
                             errorbars!(axis, [index], [mean(values)], [std(values)], [std(values)]; color = :black))
    end
    axis.xticks = (1:length(TREATMENTS), [display_names[t.name] for t in TREATMENTS])
    axis.xticklabelrotation = π / 6
end
Label(figure[0, :], "Risk-aversion treatments — mean ± 1 SD across paired runs"; fontsize = 22, font = :bold)
comparison_path = joinpath(output_dir, "risk_aversion_comparison.png")
save(comparison_path, figure; px_per_unit = 2)

distribution_figure = Figure(size = (1500, 950), fontsize = 15)
# Half-bin padding keeps the valid endpoint masses at exactly 0 and 1 visible.
bins = -0.025:0.05:1.025
for (panel, treatment) in enumerate(TREATMENTS)
    axis = Axis(distribution_figure[(panel - 1) ÷ 3 + 1, (panel - 1) % 3 + 1],
                xlabel = "risk aversion", ylabel = "probability",
                title = display_names[treatment.name])
    initial_values = [r.risk_aversion for r in distribution_rows
                      if r.treatment == treatment.name && r.phase == "initial"]
    final_values = [r.risk_aversion for r in distribution_rows
                    if r.treatment == treatment.name && r.phase == "final"]
    hist!(axis, initial_values; bins = bins, normalization = :probability,
          color = (:gray50, 0.55), label = "initial")
    hist!(axis, final_values; bins = bins, normalization = :probability,
          color = (colors[treatment.name], 0.70), label = "final")
    xlims!(axis, -0.03, 1.03)
    panel == 1 && axislegend(axis; position = :rt, framevisible = true)
end

# Unlike the pooled histograms, this panel retains the paired per-seed paths.
evolutionary_rows = sort!([row for row in rows if row.treatment == "uniform_evolutionary"];
                          by = row -> row.seed)
evolutionary_axis = Axis(distribution_figure[2, 3], xlabel = "paired seed index",
                         ylabel = "final risk mean ± within-population SD",
                         title = "Evolutionary final risk by seed")
seed_indices = 1:length(evolutionary_rows)
final_means = [row.final_risk_mean for row in evolutionary_rows]
final_sds = [row.final_risk_std for row in evolutionary_rows]
scatter!(evolutionary_axis, seed_indices, final_means; color = colors["uniform_evolutionary"],
         markersize = 10)
errorbars!(evolutionary_axis, seed_indices, final_means, final_sds, final_sds;
           color = colors["uniform_evolutionary"])
hlines!(evolutionary_axis, [0.5]; color = :black, linestyle = :dash)
xlims!(evolutionary_axis, 0.5, max(1.5, length(evolutionary_rows) + 0.5))
ylims!(evolutionary_axis, 0, 1)
Label(distribution_figure[0, :],
      "Risk distributions: pooled initial/final histograms across paired populations (30 seeds); sixth panel is per-seed";
      fontsize = 20, font = :bold)
distribution_plot_path = joinpath(output_dir, "risk_aversion_distribution.png")
save(distribution_plot_path, distribution_figure; px_per_unit = 2)
println("wrote $run_path\n$distribution_path\n$comparison_path\n$distribution_plot_path")
