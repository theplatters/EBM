function setup_world(args::ModelArgs{T}) where {T}
    world = Ark.World(
        Position,
        PrevPosition,
        Direction,
        SSensitvity,
        OSensitvity,
        Avoidance,
        Habitgene,
        Habitus,
        DriverStrategy,
        LR,
        Step,
    )
    setup_resources!(world, args)
    spawn_init_entities!(world, args.prediction_strategy)
    rebuild_occupancy!(world)
    rebuild_predicted_occupancy!(world, args.prediction_strategy)
    return world
end

function setup_world(args::ModelArgs{CapabilityModel})
    world = Ark.World(
        Position,
        PrevPosition,
        Direction,
        Speed,
        SpeedAdjustment,
        RiskAversion,
        SameDirectionResponse,
        OppositeDirectionResponse,
        NearFieldAvoidance,
        HabitFormation,
        ConventionPerception,
        PerceivedConvention,
        SocialHabitFormation,
        SocialHabitus,
        Habitus,
        LocalObservation,
        LaneScore,
        LaneProposal,
        SpeedProposal,
        MovementPath,
        LR,
        Step,
    )
    setup_resources!(world, args)
    spawn_init_entities!(world, args.prediction_strategy)
    rebuild_occupancy!(world)
    update_mean_habitus!(world)
    return world
end
