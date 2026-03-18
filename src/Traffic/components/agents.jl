@enum Direction begin
    Clockwise = 1
    Counterclockwise = -1
end

struct Step
    val::Int64
end

struct LR
    val::Float64
end

ahead_y(y::Int, dir::Direction, d::Int, h::Int) =
    mod1(y + d * Int(dir), h)
is_left_relative(x::Int, dir::Direction) =
    (dir == Clockwise) ? (x == 1) : (x == 2)
