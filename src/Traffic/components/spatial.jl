struct Ring
    width::UInt64
    height::UInt64
end

struct Position
    x::Int64
    y::Int64
end

struct PrevPosition
    x::Int64
    y::Int64
end

PrevPosition(p::Position) = PrevPosition(p.x, p.y)
Position(p::PrevPosition) = Position(p.x, p.y)
Position(t::Tuple{UInt64, UInt64}) = Position(first(t), last(t))

struct Occupancy
    grid::Matrix{Union{Nothing, Direction}}
end

function Occupancy(ring::Ring)
    return Occupancy(fill(nothing, Int(ring.width), Int(ring.height)))
end

function left(dir)
    return dir == Clockwise ? 1 : 2
end
