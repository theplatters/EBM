struct Sector
    name::String
end

struct UnitCost 
    amount::Float64
end

struct MarkUp
    amount::Float64
end

struct Quantity
    amount::Float64
end 

struct EmployedAt <: Relationship
end

struct Income
    amount::Float64
end

struct Recipient <: Relationship
end

struct LoanGiver <: Relationship
end

struct Budget
  amount::Float64
end

struct Loan 
  amount:: Float64 
  accruded_interest::Float64
end

Base.@kwdef struct BankParameters
	capital_requirement::Float64 = 0.1  # fraction of loans that bank must keep as capital
  dividend_share::Float64 = 0.4
  interest_rate::Float64 = 0.05
end

Base.@kwdef struct CentralBankParams
	interest_rate::Float64 = 0.01   # policy rate used for central bank lending
end

Base.@kwdef struct IndustryParameters
	deprecation_rate::Float64 = 0.05
	dividend_share::Float64 = 0.5      # fraction of profits paid as dividends
	investment_cost::Float64 = 1.0      # cost per unit of capital (in price units)
	max_markup::Float64 = 0.3
	min_markup::Float64 = 0.05
end

Base.@kwdef struct HouseholdParameters
	consumption_share::Vector{Float64}   # fraction across goods (sum=1)
	income_propensity::Float64 = 0.8            # α1 (propensity to consume out of dividends)
	wealth_propensity::Float64 = 0.1            # α2 (propensity to consume out of wealth)
end

struct GovernmentParameters
	tax_adjustments::Vector{Pair{Float64, Float64}}  # (debt_to_gdp_threshold => tax_adjustment_factor)

	function GovernmentParameters(tax_adjustments = nothing)
		if tax_adjustments === nothing
			# Default tax adjustment schedule based on debt-to-GDP ratios
			tax_adjustments = [
				0.3 => 0.8,   # Low debt: reduce taxes
				0.6 => 1.0,   # Moderate debt: maintain taxes
				0.9 => 1.2,   # High debt: increase taxes
				1.2 => 1.5,    # Very high debt: significantly increase taxes
			]
		end
		new(tax_adjustments)
	end
end

struct InputOutputTable
    technical_coefficients::Matrix{Float64}  # technical coefficients (input-output matrix) (Ngoods x Ngoods)
    leontief::LU{Float64, Matrix{Float64}}   # Leontief inverse (Ngoods x Ngoods)
end


function accrude_interest!(world)
end

function form_final_demand_expectations!(world)
end

function produce_goods!(world)
end

function set_prices!(world)
end

function formulate_nominal_demand!(world)
end

function sell_products!(world)
end

function calculate_investment_need!(world)
end

function give_out_loans!(world)
end

function order_investments!(world)
end

function taxes!(world)
end

function dividends!(world)
end

function payback_loans!(world)
end

function request_central_bank_loans_to_banks!(world)
end

function request_central_bank_loans_to_government!(world)
end

