Revise.retry()
using EBM
using GLMakie
logger = EBM.Traffic.main(Dict())


to_heatmap(lp) = Int.(replace(lp, nothing => 0))


EBM.Traffic.plot_torus(logger.positions[end] |> to_heatmap)
