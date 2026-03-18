function to_point_on_circle(x, y, R_big = 1.0, R_small = 0.7)
    α =  2 * π * y / 100
    R = x == 1 ? R_small : R_big
    return R * [sin(α), cos(α)]
end

function plot_torus(grid::Vector{Matrix{Int}})

    f = Figure(size=(1000,1000))
    ax = Axis(f[1, 1])

    
    color_map = Dict(
    0=> RGBf(1,1,1),
    -1 => RGBf(1,0,0),
    1 => RGBf(0,1,0),
  )

    arc!(ax, Point2f(0), 1, -π, π, color = :black)
    arc!(ax, Point2f(0), 0.7, -π, π, color = :black)

    step = Observable(1)
    x = @lift [to_point_on_circle(index[1],index[2])[1]  for index in CartesianIndices(grid[$step])] |> vec
    y = @lift [to_point_on_circle(index[1],index[2])[2]  for index in CartesianIndices(grid[$step])] |> vec
    colors = @lift [color_map[val] for val in grid[$step]] |> vec

    @info colors

    scatter!(ax,x,y, color = colors, markersize = 30)


    framerate = 5
    timestamps = 1:length(grid) 

    i = 1
    record(f, "time_animation.mkv", timestamps;
        framerate = framerate) do t
    step[] = i
    i += 1
    end
    return f

end
