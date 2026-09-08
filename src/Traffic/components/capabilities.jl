abstract type CapabilityReplacementPolicy end

"""Draw every replacement independently from the configured entry shares."""
struct EntryDrawReplacement <: CapabilityReplacementPolicy end

"""
Inherit a uniformly selected survivor's capability genome, with mutation.

`capability_mutation_rate` is the independent probability that each optional
capability is gained or lost. `trait_mutation_scale` is the Gaussian standard
deviation applied to inherited continuous trait values.
"""
Base.@kwdef struct EvolutionaryReplacement <: CapabilityReplacementPolicy
    capability_mutation_rate::Float64 = 0.02
    trait_mutation_scale::Float64 = 0.05
end

function validate(policy::EvolutionaryReplacement)
    0.0 <= policy.capability_mutation_rate <= 1.0 ||
        throw(ArgumentError("capability_mutation_rate must be in [0, 1]"))
    policy.trait_mutation_scale >= 0.0 ||
        throw(ArgumentError("trait_mutation_scale must be nonnegative"))
    return policy
end

validate(policy::EntryDrawReplacement) = policy

"""
Configuration for the bounded-information, capability-composed traffic model.

Capability shares are independent probabilities, so cars may carry any
meaningful combination of behavioral mechanisms. Every car has speed control;
the optional components determine how it evaluates lanes and conventions.
"""
Base.@kwdef struct CapabilityModel <: OccupancyStrategy
    same_direction_share::Float64 = 0.75
    opposite_direction_share::Float64 = 0.75
    avoidance_share::Float64 = 0.75
    habit_share::Float64 = 0.5
    convention_share::Float64 = 0.5
    social_habit_share::Float64 = 0.0
    habit_weight::Float64 = 0.5
    convention_weight::Float64 = 0.5
    social_habit_weight::Float64 = 0.5
    convention_learning_rate::Float64 = 0.2
    convention_noise::Float64 = 0.05
    social_habit_learning_rate::Float64 = 0.2
    social_habit_noise::Float64 = 0.05
    social_trace_retention::Float64 = 0.9
    social_trace_deposit::Float64 = 0.25
    max_speed::Int = 3
    avoidance_disable_age::Union{Nothing, Int} = nothing
    replacement_policy::CapabilityReplacementPolicy = EntryDrawReplacement()
end

function validate(model::CapabilityModel)
    shares = (
        model.same_direction_share,
        model.opposite_direction_share,
        model.avoidance_share,
        model.habit_share,
        model.convention_share,
        model.social_habit_share,
    )
    all(share -> 0.0 <= share <= 1.0, shares) ||
        throw(ArgumentError("capability shares must be in [0, 1]"))
    model.habit_weight >= 0.0 || throw(ArgumentError("habit_weight must be nonnegative"))
    model.convention_weight >= 0.0 ||
        throw(ArgumentError("convention_weight must be nonnegative"))
    model.social_habit_weight >= 0.0 ||
        throw(ArgumentError("social_habit_weight must be nonnegative"))
    0.0 <= model.convention_learning_rate <= 1.0 ||
        throw(ArgumentError("convention_learning_rate must be in [0, 1]"))
    model.convention_noise >= 0.0 ||
        throw(ArgumentError("convention_noise must be nonnegative"))
    0.0 <= model.social_habit_learning_rate <= 1.0 ||
        throw(ArgumentError("social_habit_learning_rate must be in [0, 1]"))
    model.social_habit_noise >= 0.0 ||
        throw(ArgumentError("social_habit_noise must be nonnegative"))
    0.0 <= model.social_trace_retention < 1.0 ||
        throw(ArgumentError("social_trace_retention must be in [0, 1)"))
    model.social_trace_deposit > 0.0 ||
        throw(ArgumentError("social_trace_deposit must be positive"))
    1 <= model.max_speed <= 3 || throw(ArgumentError("max_speed must be in 1:3"))
    isnothing(model.avoidance_disable_age) || model.avoidance_disable_age > 0 ||
        throw(ArgumentError("avoidance_disable_age must be positive"))
    validate(model.replacement_policy)
    return model
end

"""Current integer speed. Capability cars always move at least one cell."""
struct Speed
    val::Int

    function Speed(value::Integer)
        1 <= value <= 3 || throw(ArgumentError("speed must be in 1:3"))
        return new(Int(value))
    end
end

struct SameDirectionResponse
    sensitivity::Float64
end

struct OppositeDirectionResponse
    sensitivity::Float64
end

struct NearFieldAvoidance
    sensitivity::Float64
end

"""Hodgson–Knudsen capability for reinforcing the driver's own realized side."""
struct HabitFormation
    disposition::Float64
end

"""Capability for remembering the locally observed side choices of other drivers."""
struct ConventionPerception
    learning_rate::Float64
    noise::Float64
end

"""Private disposition calculated from a history of local lane-choice observations."""
struct PerceivedConvention
    value::Float64
    confidence::Float64
end

"""Capability for building a lane disposition from successful-driver traces."""
struct SocialHabitFormation
    disposition::Float64
    learning_rate::Float64
    noise::Float64
end

"""Private lane disposition built from a history of observed success traces."""
struct SocialHabitus
    value::Float64
end

struct SpeedAdjustment
    max_speed::Int
end

struct RiskAversion
    value::Float64
    function RiskAversion(value::Real)
        value = Float64(value)
        0.0 <= value <= 1.0 || throw(ArgumentError("risk aversion must be in [0, 1]"))
        new(value)
    end
end

"""Private observation copied from the committed state at the start of a tick."""
struct LocalObservation
    same_left::Float64
    opposite_left::Float64
    close_left::Float64
    close_right::Float64
    convention::Float64
    convention_samples::Int
    success_trace::Float64
    success_trace_samples::Int
end

LocalObservation(same_left, opposite_left, close_left, close_right, convention, convention_samples) =
    LocalObservation(
        same_left, opposite_left, close_left, close_right,
        convention, convention_samples, 0.0, 0,
    )

LocalObservation() = LocalObservation(0.5, 0.5, 0.0, 0.0, 0.0, 0, 0.0, 0)

struct LaneScore
    value::Float64
end

struct LaneProposal
    lane::Int
end

struct SpeedProposal
    value::Int
end

"""The three micro-step positions implied by a car's private proposal."""
struct MovementPath
    positions::NTuple{3, Position}
end

MovementPath(position::Position) = MovementPath((position, position, position))

"""Heritable risk aversion and optional components, excluding acquired state."""
struct CapabilityGenome
    risk_aversion::Float64
    same_direction::Union{Nothing, Float64}
    opposite_direction::Union{Nothing, Float64}
    avoidance::Union{Nothing, Float64}
    habit::Union{Nothing, Float64}
    convention::Union{Nothing, ConventionPerception}
    social_habit::Union{Nothing, SocialHabitFormation}
end

const CAPABILITY_COMPONENTS = (
    SameDirectionResponse,
    OppositeDirectionResponse,
    NearFieldAvoidance,
    HabitFormation,
    ConventionPerception,
    SocialHabitFormation,
)

function capability_mask(world, entity)
    mask = UInt8(0)
    for (index, component) in enumerate(CAPABILITY_COMPONENTS)
        Ark.has_components(world, entity, (component,)) &&
            (mask |= UInt8(1) << (index - 1))
    end
    return mask
end
