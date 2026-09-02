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

struct Disease
    bits::UInt64
    length::Int64

    function Disease(bits::Integer, length::Integer)
        1 <= length <= 64 || throw(ArgumentError("disease length must be between 1 and 64"))
        mask = length == 64 ? typemax(UInt64) : (UInt64(1) << length) - UInt64(1)
        return new(UInt64(bits) & mask, Int64(length))
    end
end

"""Inherited immune template and the phenotype trained during the citizen's life."""
struct ImmuneProfile
    genotype::UInt64
    phenotype::UInt64
end

ImmuneProfile(bits::Integer) = ImmuneProfile(UInt64(bits), UInt64(bits))

"""Bit mask of diseases carried from the model's shared disease catalogue."""
struct Infection
    diseases::UInt64

    function Infection(diseases::Integer)
        diseases >= 0 || throw(ArgumentError("infection mask cannot be negative"))
        return new(UInt64(diseases))
    end
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
