function decide_attendance!(world)
    capacity = Ark.get_resource(world, ModelParams).capacity
    for (entities, forecasts, decisions) in
        Query(world, (ExpectedAttendance, AttendanceDecision))
        @inbounds for i in eachindex(entities)
            decisions[i] = AttendanceDecision(forecasts[i].val < capacity)
        end
    end
    return nothing
end

function aggregate_attendance!(world)
    attendance = 0
    for (entities, decisions) in Query(world, (AttendanceDecision,))
        @inbounds for i in eachindex(entities)
            attendance += decisions[i].attend
        end
    end
    Ark.get_resource(world, CurrentAttendance).val = attendance
    return nothing
end

function update_payoffs!(world)
    params = Ark.get_resource(world, ModelParams)
    attendance = Ark.get_resource(world, CurrentAttendance).val
    uncrowded = attendance < params.capacity

    for (entities, decisions, payoffs) in
        Query(world, (AttendanceDecision, CumulativePayoff))
        @inbounds for i in eachindex(entities)
            increment = if decisions[i].attend && uncrowded
                params.successful_attendance_payoff
            elseif !decisions[i].attend && !uncrowded
                params.successful_absence_payoff
            else
                0.0
            end
            payoffs[i] = CumulativePayoff(payoffs[i].val + increment)
        end
    end
    return nothing
end
