using EBM

output_dir = joinpath(@__DIR__, "plots")
params = EBM.Sugarscape.ModelParams(
    disease_catalog_size = 10,
    initial_diseases_per_citizen = 4,
)
args = EBM.Sugarscape.ModelArgs(seed = 2026, params = params, steps = 250)
result = EBM.Sugarscape.generate_plot_suite(args; output_dir = output_dir)

println("Generated Sugarscape visualization suite:")
foreach(path -> println("  ", path), result.paths)
println("Final Gini coefficient: ", round(result.statistics.gini; digits = 3))
println(
    "Top 10% wealth share: ",
    round(100 * result.statistics.top_10_wealth_share; digits = 1),
    "%",
)
