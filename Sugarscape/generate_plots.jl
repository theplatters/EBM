using EBM

output_dir = joinpath(@__DIR__, "plots")
args = EBM.Sugarscape.ModelArgs(seed = 2026, steps = 250)
result = EBM.Sugarscape.generate_plot_suite(args; output_dir = output_dir)

println("Generated Sugarscape plots:")
foreach(path -> println("  ", path), result.paths)
