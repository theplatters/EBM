using EBM
using Ark
using Test

const ElFarol = EBM.ElFasol

function count_components(world, component_types)
    return sum(length(entities) for (entities, _...) in Ark.Query(world, component_types))
end

@testset "El Farol setup" begin
    params = ElFarol.ModelParams(
        population = 25,
        capacity = 15,
        predictors_per_agent = 6,
        history_length = 12,
    )
    world = ElFarol.setup_world(ElFarol.ModelArgs(seed = 11, params = params, steps = 0))

    @test count_components(world, (ElFarol.ParticipantId,)) == params.population
    @test count_components(world, (ElFarol.PredictorId,)) ==
          params.population * params.predictors_per_agent
    history = Ark.get_resource(world, ElFarol.AttendanceHistory)
    @test length(history.values) == params.history_length
    @test all(0 <= attendance <= params.population for attendance in history.values)
end

@testset "El Farol predictor semantics" begin
    history = Int64[40, 50, 60, 70]
    @test ElFarol.evaluate_predictor(ElFarol.LagPredictor(2), history, 100) == 60.0
    @test ElFarol.evaluate_predictor(ElFarol.MeanPredictor(2), history, 100) == 65.0
    @test ElFarol.evaluate_predictor(ElFarol.MirrorPredictor(50.0, 1), history, 100) ==
          30.0
    @test ElFarol.evaluate_predictor(ElFarol.TrendPredictor(4), history, 100) == 80.0
    @test ElFarol.evaluate_predictor(ElFarol.ConstantPredictor(150.0), history, 100) ==
          100.0
end

@testset "El Farol synchronous step and reproducibility" begin
    params = ElFarol.ModelParams(
        population = 40,
        capacity = 24,
        predictors_per_agent = 8,
        history_length = 15,
    )
    args = ElFarol.ModelArgs(seed = 2026, params = params, steps = 50)
    first = ElFarol.main(args)
    second = ElFarol.main(args)

    @test first.attendance == second.attendance
    @test first.mean_forecast == second.mean_forecast
    @test first.forecast_std == second.forecast_std
    @test first.predictor_shares == second.predictor_shares
    @test first.predictor_rmse == second.predictor_rmse
    @test length(first.attendance) == args.steps
    @test all(0 <= attendance <= params.population for attendance in first.attendance)
    @test all(0.0 <= rate <= 1.0 for rate in first.attendance_rate)
    @test all(isapprox(sum(shares), 1.0) for shares in first.predictor_shares)
    @test all(
        all(isfinite(error) for error in family_errors)
            for family_errors in first.predictor_rmse
    )

    world = ElFarol.setup_world(
        ElFarol.ModelArgs(
            seed = 7,
            params = params,
            steps = 0,
            initial_history = fill(Int64(20), params.history_length),
        ),
    )
    ElFarol.step!(world)
    attendance = Ark.get_resource(world, ElFarol.CurrentAttendance).val
    history = Ark.get_resource(world, ElFarol.AttendanceHistory).values
    @test last(history) == attendance
    @test length(history) == params.history_length + 1

    scored_predictors = 0
    for (entities, forecasts, errors) in
        Ark.Query(world, (ElFarol.Forecast, ElFarol.SquaredError))
        @inbounds for i in eachindex(entities)
            @test errors[i].val == (forecasts[i].val - attendance)^2
            scored_predictors += 1
        end
    end
    @test scored_predictors == params.population * params.predictors_per_agent

    selected = Int64[]
    for (_, predictor_ids) in Ark.Query(world, (ElFarol.SelectedPredictor,))
        append!(selected, predictor_id.val for predictor_id in predictor_ids)
    end
    @test length(selected) == params.population
    @test all(!=(0), selected)
end

@testset "El Farol plotting diagnostics" begin
    params = ElFarol.ModelParams(population = 20, capacity = 12, predictors_per_agent = 5)
    logger = ElFarol.main(ElFarol.ModelArgs(seed = 31, params = params, steps = 12))

    @test !isnothing(ElFarol.plot_attendance_dynamics(logger, params; rolling_window = 4))
    @test !isnothing(
        ElFarol.plot_coordination_diagnostics(logger, params; burn_in = 3, maximum_lag = 5),
    )
    @test !isnothing(ElFarol.plot_predictor_ecology(logger; averaging_window = 4))
    @test_throws ArgumentError ElFarol.plot_attendance_dynamics(ElFarol.Logger(), params)
end

@testset "El Farol validation" begin
    @test_throws ArgumentError ElFarol.setup_world(
        ElFarol.ModelArgs(params = ElFarol.ModelParams(population = 10, capacity = 11)),
    )
    @test_throws ArgumentError ElFarol.setup_world(
        ElFarol.ModelArgs(initial_history = Int64[1]),
    )
end
