struct Sector
    name::String
end

struct Interest
    amount::Float64
end

struct Price
    amount::Float64
end

struct UnitCost
    amount::Float64
end

struct MarkUp
    amount::Float64
end


struct InputOutputTable
    technical_coefficients::Matrix{Float64}  # technical coefficients (input-output matrix) (Ngoods x Ngoods)
    leontief::LU{Float64, Matrix{Float64}}   # Leontief inverse (Ngoods x Ngoods)
end
