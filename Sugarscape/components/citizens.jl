struct CitizenId
    val::Int64
end

struct Position
    x::Int64
    y::Int64
end

struct ProposedPosition
    x::Int64
    y::Int64
end

ProposedPosition(position::Position) = ProposedPosition(position.x, position.y)
Position(proposal::ProposedPosition) = Position(proposal.x, proposal.y)

struct Vision
    val::Int64
end

struct Metabolism
    val::Int64
end

struct Sugar
    val::Int64
end

struct Age
    val::Int64
end

struct MaximumAge
    val::Int64
end

struct InitialEndowment
    val::Int64
end

struct Female end
struct Male end

struct ImmuneProfile
    bits::UInt64
end

struct Infection
    strain::UInt64
    age::Int64
end

struct CitizenState
    id::Int64
    position::Position
    vision::Int64
    metabolism::Int64
    sugar::Int64
    age::Int64
    maximum_age::Int64
    sex::Symbol
    infected::Bool
end
