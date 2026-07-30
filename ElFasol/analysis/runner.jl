function run_model(args::ModelArgs = ModelArgs())
    world = setup_world(args)
    for _ in 1:args.steps
        step!(world)
    end
    return world
end

function main(args::ModelArgs = ModelArgs())
    world = run_model(args)
    return Ark.get_resource(world, Logger)
end
