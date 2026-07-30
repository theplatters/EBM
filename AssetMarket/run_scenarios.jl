using EBM

output_dir = joinpath(@__DIR__, "scenarios")
result = EBM.AssetMarket.generate_scenario_analysis(
    EBM.AssetMarket.default_scenarios();
    output_dir = output_dir,
    steps = 1_500,
    burn_in = 300,
    replicates = 5,
    seed = 2026,
)

println("Generated asset-market scenario analysis:")
foreach(path -> println("  ", path), result.tables)
foreach(path -> println("  ", path), result.paths)
