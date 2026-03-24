using GLM, DataFrames

function weight_regression(res::Vector{SweepResult})
    df = DataFrame(
        ws = [el.weights.wₛ for el in res],
        wo = [el.weights.wₒ for el in res],
        wa = [el.weights.wₐ for el in res],
        wh = [el.weights.wₕ for el in res],
        output = [mean(el.logger.mean_abs_habitus[(end - 5):end]) for el in res]
    )

    # Drop one weight to avoid perfect multicollinearity (sum-to-one constraint)
    model = lm(@formula(output ~ ws + wo + wh + ws^2 + wo^2 + wh^2 + ws * wh + wo * wh), df)

    println(model)
    return model
end
function weight_regression_age(res::Vector{SweepResult})
    df = DataFrame(
        ws = [el.weights.wₛ for el in res],
        wo = [el.weights.wₒ for el in res],
        wa = [el.weights.wₐ for el in res],
        wh = [el.weights.wₕ for el in res],
        output = [mean(el.logger.mean_age) for el in res]
    )

    # Drop one weight to avoid perfect multicollinearity (sum-to-one constraint)
    model = lm(@formula(output ~ ws + wo + wh + ws^2 + wo^2 + wh^2 + ws * wh + wo * wh), df)

    println(model)
    return model
end
