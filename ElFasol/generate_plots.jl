using EBM

output_dir = joinpath(@__DIR__, "plots")
args = EBM.ElFasol.ModelArgs(seed = 2026, steps = 750)
result = EBM.ElFasol.generate_plot_suite(
    args;
    output_dir = output_dir,
    burn_in = 150,
    rolling_window = 25,
)

println("Generated El Farol plots:")
foreach(path -> println("  ", path), result.paths)
