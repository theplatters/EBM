function growback!(world)
  rate = Ark.get_resource(world, ModelParams).growback_rate
  iszero(rate) && return nothing
  landscape = Ark.get_resource(world, SugarLandscape)
  @inbounds for index in eachindex(landscape.current)
    landscape.current[index] = min(
      landscape.capacity[index],
      landscape.current[index] + rate,
    )
  end
  return nothing
end
