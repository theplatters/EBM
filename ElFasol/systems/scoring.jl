function update_predictor_scores!(world)
    attendance = Ark.get_resource(world, CurrentAttendance).val
    for (entities, forecasts, errors) in Query(world, (Forecast, SquaredError))
        @inbounds for i in eachindex(entities)
            error = forecasts[i].val - attendance
            errors[i] = SquaredError(errors[i].val + error^2)
        end
    end
    return nothing
end

function finalize_period!(world)
    attendance = Ark.get_resource(world, CurrentAttendance).val
    push!(Ark.get_resource(world, AttendanceHistory).values, attendance)
    Ark.get_resource(world, SimulationClock).step += 1
    return nothing
end
