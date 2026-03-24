function to_point_on_circle(x, y, R_big = 1.0, R_small = 0.7)
    α = 2 * π * y / 100
    R = x == 1 ? R_small : R_big
    return R * [sin(α), cos(α)]
end

function plot_torus(grid::Vector{Matrix{Int}})

    f = Figure(size = (1000, 1000))
    ax = Axis(f[1, 1])


    color_map = Dict(
        0 => RGBf(1, 1, 1),
        -1 => RGBf(1, 0, 0),
        1 => RGBf(0, 1, 0),
    )

    arc!(ax, Point2f(0), 1, -π, π, color = :black)
    arc!(ax, Point2f(0), 0.7, -π, π, color = :black)

    step = Observable(1)
    x = @lift [to_point_on_circle(index[1], index[2])[1]  for index in CartesianIndices(grid[$step])] |> vec
    y = @lift [to_point_on_circle(index[1], index[2])[2]  for index in CartesianIndices(grid[$step])] |> vec
    colors = @lift [color_map[val] for val in grid[$step]] |> vec

    @info colors

    scatter!(ax, x, y, color = colors, markersize = 30)


    framerate = 5
    timestamps = 1:length(grid)

    i = 1
    record(
        f, "time_animation.mkv", timestamps;
        framerate = framerate
    ) do t
        step[] = i
        i += 1
    end
    return f

end

function plot_logger(logger::AbstractLogger)
    n = length(logger.mean_habitus)
    x = 1:n

    f = Figure()

    # --- First axis: mean_habitus + mean_abs_habitus (two y-axes) ---
    ax1 = Axis(f[1, 1], ylabel = "mean_habitus", xlabel = "step")
    ax1r = Axis(f[1, 1], yaxisposition = :right, ylabel = "mean_abs_habitus")

    # Hide duplicate x decorations on right axis
    hidexdecorations!(ax1r)

    # Link x axes
    linkxaxes!(ax1, ax1r)

    lines!(ax1, x, logger.mean_habitus, label = "mean_habitus")
    lines!(ax1r, x, logger.mean_abs_habitus, color = :red, label = "mean_abs_habitus")

    axislegend(ax1, position = :lt)

    # --- Second axis: mean_age + deaths (two y-axes) ---
    ax2 = Axis(f[2, 1], ylabel = "mean_age", xlabel = "step")
    ax2r = Axis(f[2, 1], yaxisposition = :right, ylabel = "deaths")

    hidexdecorations!(ax2r)

    linkxaxes!(ax2, ax2r)

    lines!(ax2, x, logger.mean_age, label = "mean_age")
    lines!(ax2r, x, logger.deaths, color = :orange, label = "deaths")

    axislegend(ax2, position = :lt)

    return f
end


function animate_densities(logger::AbstractLogger, framerate = 5)
    step = Observable(1)

    f = Figure()
    ax = Axis(f[1, 1])

    s = @lift logger.distribution_S[$step]
    o = @lift logger.distribution_O[$step]
    a = @lift logger.distribution_A[$step]

    density!(ax, s)
    density!(ax, o)
    density!(ax, a)

    timestamps = 1:length(logger.distribution_A)
    record(
        f, "density_animation.mkv", timestamps;
        framerate = framerate
    ) do t
        step[] = t
    end

    return nothing
end


function plot_parallell_coordinates(res::Vector{SweepResult})
    mapped = map(res) do el
        [
            el.weights.wₛ, el.weights.wₒ, el.weights.wₐ, el.weights.wₕ,
            mean(el.logger.mean_age),
        ]
    end

    output_vals = [row[5] for row in mapped]
    lo, hi = minimum(output_vals), maximum(output_vals)
    normalize = v -> (v - lo) / (hi - lo + eps())

    map(mapped) do (v)
        v[5] = v[5] / hi
    end

    cmap = cgrad(:viridis)
    colors = [cmap[normalize(row[5])] for row in mapped]

    sort!(mapped)
    fig = Figure()
    ax = Axis(
        fig[1, 1],
        xticks = (1:5, ["wₛ", "wₒ", "wₐ", "wₕ", "habitus"]),
        ylabel = "value"
    )
    step = Observable(1)


    row = @lift mapped[$step]
    color = @lift colors[$step]

    lines!(ax, 1:5, row; color = color, linewidth = 1.2, alpha = 0.6)

    Colorbar(
        fig[1, 2];
        colormap = :viridis,
        limits = (0, 1),
        label = "mean abs habitus"
    )


    timestamps = 1:length(mapped)
    record(
        fig, "krass.mkv", timestamps;
        framerate = 10
    ) do t
        step[] = t
    end


    return fig
end
