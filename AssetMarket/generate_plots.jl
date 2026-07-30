using EBM

output_dir = joinpath(@__DIR__, "plots")
args = EBM.AssetMarket.ModelArgs(seed = 2026, steps = 2_500)
result = EBM.AssetMarket.generate_plot_suite(
    args;
    output_dir = output_dir,
    burn_in = 500,
    rolling_window = 50,
)

println("Generated asset-market plots:")
foreach(path -> println("  ", path), result.paths)
