# find indices of all df row where f is true
findallrows(f::Function, df) = findall(f(r) for r in eachrow(df))

# extract parameter estimates from the lavaan result
function lav_parameters(partable_lav, partable::ParameterTable,
                        params::Vector{Symbol},
                        lav_col::Symbol = :est, lav_group = nothing)
    partable_mask = findall(partable.columns[:free])
    param_index = Dict(param => i for (i, param) in enumerate(params))
    res = fill(NaN, length(param_index))

    lav_values = partable_lav[:, lav_col]
    for (from, to, type, id) in
        zip([view(partable.columns[k], partable_mask)
             for k in [:from, :to, :parameter_type, :identifier]]...)

        lav_ind = nothing

        if from == Symbol("1")
            lav_ind = findallrows(r -> r[:lhs] == String(to) && r[:op] == "~1" &&
                                  (isnothing(lav_group) || r[:group] == lav_group), partable_lav)
        else
            if type == :↔
                lav_type = "~~"
            elseif type == :→
                if (from ∈ partable.variables[:latent_vars]) && (to ∈ partable.variables[:observed_vars])
                    lav_type = "=~"
                else
                    lav_type = "~"
                    from, to = to, from
                end
            end

            if lav_type == "~~"
                lav_ind = findallrows(r -> ((r[:lhs] == String(from) && r[:rhs] == String(to)) ||
                                        (r[:lhs] == String(to) && r[:rhs] == String(from))) &&
                                        r[:op] == lav_type &&
                                        (isnothing(lav_group) || r[:group] == lav_group),
                                      partable_lav)
            else
                lav_ind = findallrows(r -> r[:lhs] == String(from) && r[:rhs] == String(to) && r[:op] == lav_type &&
                                      (isnothing(lav_group) || r[:group] == lav_group),
                                      partable_lav)
            end
        end

        if length(lav_ind) == 0
            throw(ErrorException("Parameter $id ($from $type $to) could not be found in the lavaan solution"))
        elseif length(lav_ind) > 1
            throw(ErrorException("At least one parameter was found twice in the lavaan solution"))
        end

        res_ind = param_index[id]
        res_val = lav_values[lav_ind[1]]
        if isnan(res[res_ind])
            res[res_ind] = res_val
        else
            @test res[res_ind] ≈ res_val atol=1E-10
        end
    end

    return res
end

function test_gradient(model, parameters; rtol = 1e-10, atol = 0)
    true_grad = FiniteDiff.finite_difference_gradient(x -> objective!(model, x)[1], parameters)
    gradient = similar(parameters); gradient .= 1.0

    # F and G
    gradient!(gradient, model, parameters)
    correct1 = isapprox(gradient, true_grad; rtol = rtol, atol = atol)

    # only G
    gradient .= 1.0
    objective_gradient!(gradient, model, parameters)
    correct2 = isapprox(gradient, true_grad; rtol = rtol, atol = atol)

    return correct1 & correct2
end

function test_hessian(model, parameters; rtol = 1e-4, atol = 0)
    true_hessian = FiniteDiff.finite_difference_hessian(x -> objective!(model, x)[1], parameters)
    hessian = zeros(size(true_hessian)); hessian .= 1.0
    gradient = similar(parameters)

    # H
    hessian!(hessian, model, parameters)
    correct1 = isapprox(hessian, true_hessian; rtol = rtol, atol = atol)

    # F and H
    hessian .= 1.0
    objective_hessian!(hessian, model, parameters)
    correct2 = isapprox(hessian, true_hessian; rtol = rtol, atol = atol)

    # G and H
    hessian .= 1.0
    gradient_hessian!(gradient, hessian, model, parameters)
    correct3 = isapprox(hessian, true_hessian; rtol = rtol, atol = atol)

    # F, G and H
    hessian .= 1.0
    objective_gradient_hessian!(gradient, hessian, model, parameters)
    correct4 = isapprox(hessian, true_hessian; rtol = rtol, atol = atol)

    return correct1 & correct2 & correct3 & correct4
end

fitmeasure_names_ml = Dict(
    :AIC => "aic",
    :BIC => "bic",
    :df => "df",
    :χ² => "chisq",
    :p_value => "pvalue",
    :n_par => "npar",
    :RMSEA => "rmsea",
)

fitmeasure_names_ls = Dict(
    :df => "df",
    :χ² => "chisq",
    :p_value => "pvalue",
    :n_par => "npar",
    :RMSEA => "rmsea",
)

function test_fitmeasures(
        measures,
        measures_lav;
        rtol = 1e-4,
        atol = 0,
        fitmeasure_names = fitmeasure_names_ml)
    correct = []
    for key in keys(fitmeasure_names)
        measure = measures[key]
        measure_lav = measures_lav.x[measures_lav[:, 1] .==  fitmeasure_names[key]][1]
        push!(correct, isapprox(measure, measure_lav; rtol = rtol, atol = atol))
    end
    return correct
end

function compare_estimates(partable::ParameterTable, partable_lav;
        rtol = 1e-10, atol = 0, col = :estimate, lav_col = :est)

    estimates = Vector{Float64}()
    expected = Vector{Float64}()
    correct = Bool[]

    for i in findall(partable.columns[:free])

        from = partable.columns[:from][i]
        to = partable.columns[:to][i]
        type = partable.columns[:parameter_type][i]
        estimate = partable.columns[col][i]
        push!(estimates, estimate)

        if from == Symbol("1")

            lav_ind = findall(
                (partable_lav.lhs .== String(to)) .&
                (partable_lav.op .== "~1"))

            if length(lav_ind) == 0
                throw(ErrorException("Parameter from: $from, to: $to, type: $type, could not be found in the lavaan solution"))
            elseif length(lav_ind) > 1
                throw(ErrorException("At least one parameter was found twice in the lavaan solution"))
            else
                is_correct = isapprox(
                    estimate,
                    partable_lav[:, lav_col][lav_ind[1]];
                    rtol = rtol,
                    atol = atol)
                push!(expected, partable_lav[:, lav_col][lav_ind[1]])
                push!(correct, is_correct)
            end

        else

            if type == :↔
                type = "~~"
            elseif type == :→
                if (from ∈ partable.variables[:latent_vars]) & (to ∈ partable.variables[:observed_vars])
                    type = "=~"
                else
                    type = "~"
                    from, to = to, from
                end
            end

            if type == "~~"

                lav_ind = findall(
                    (
                        ((partable_lav.lhs .== String(from)) .& (partable_lav.rhs .== String(to))) .|
                        ((partable_lav.lhs .== String(to)) .& (partable_lav.rhs .== String(from)))
                    ) .&
                    (partable_lav.op .== type)
                )

                if length(lav_ind) == 0
                    throw(ErrorException("Parameter from: $from, to: $to, type: $type, could not be found in the lavaan solution"))
                elseif length(lav_ind) > 1
                    throw(ErrorException("At least one parameter was found twice in the lavaan solution"))
                else
                    is_correct = isapprox(
                        estimate,
                        partable_lav[:, lav_col][lav_ind[1]];
                        rtol = rtol,
                        atol = atol)
                    push!(expected, partable_lav[:, lav_col][lav_ind[1]])
                    push!(correct, is_correct)
                end

            else
                lav_ind = findall(
                    (partable_lav.lhs .== String(from)) .&
                    (partable_lav.rhs .== String(to)) .&
                    (partable_lav.op .== type))

                if length(lav_ind) == 0
                    throw(ErrorException("Parameter from: $from, to: $to, type: $type, could not be found in the lavaan solution"))
                elseif length(lav_ind) > 1
                    throw(ErrorException("At least one parameter was found twice in the lavaan solution"))
                else
                    is_correct = isapprox(estimate, partable_lav[:, lav_col][lav_ind[1]]; rtol = rtol, atol = atol)
                    push!(expected, partable_lav[:, lav_col][lav_ind[1]])
                    push!(correct, is_correct)
                end
            end

        end

    end

    @show estimates - expected maximum(abs, estimates - expected)
    @test estimates ≈ expected atol=atol rtol=rtol norm=Base.Fix2(norm, Inf)
    @testset "v[$i]" for i in eachindex(expected)
        @test estimates[i] ≈ expected[i] atol=atol rtol=rtol norm=Base.Fix2(norm, Inf)
    end
    return all(correct)
end

function compare_estimates(ens_partable::EnsembleParameterTable, partable_lav;
    rtol = 1e-10, atol = 0, col = :estimate, lav_col = :est,
    lav_groups)

    correct = Bool[]

    for key in keys(ens_partable.tables)

        group = lav_groups[key]
        partable = ens_partable.tables[key]

        for i in findall(partable.columns[:free])

            from = partable.columns[:from][i]
            to = partable.columns[:to][i]
            type = partable.columns[:parameter_type][i]
            estimate = partable.columns[col][i]

            if from == Symbol("1")

                lav_ind = findall(
                    (partable_lav.lhs .== String(to)) .&
                    (partable_lav.op .== "~1") .&
                    (partable_lav.group .== group))

                if length(lav_ind) == 0
                    throw(ErrorException("Mean parameter of variable $to could not be found in the lavaan solution"))
                elseif length(lav_ind) > 1
                    throw(ErrorException("At least one parameter was found twice in the lavaan solution"))
                else
                    is_correct = isapprox(
                        estimate,
                        partable_lav[:, lav_col][lav_ind[1]];
                        rtol = rtol,
                        atol = atol)
                    push!(correct, is_correct)
                end

            else

                if type == :↔
                    type = "~~"
                elseif type == :→
                    if (from ∈ partable.variables[:latent_vars]) & (to ∈ partable.variables[:observed_vars])
                        type = "=~"
                    else
                        type = "~"
                        from, to = to, from
                    end
                end

                if type == "~~"

                    lav_ind = findall(
                        (
                            ((partable_lav.lhs .== String(from)) .& (partable_lav.rhs .== String(to))) .|
                            ((partable_lav.lhs .== String(to)) .& (partable_lav.rhs .== String(from)))
                        ) .&
                        (partable_lav.op .== type) .&
                        (partable_lav.group .== group)
                    )

                    if length(lav_ind) == 0
                        throw(ErrorException("Parameter from: $from, to: $to, type: $type, could not be found in the lavaan solution"))
                    elseif length(lav_ind) > 1
                        throw(ErrorException("At least one parameter was found twice in the lavaan solution"))
                    else
                        is_correct = isapprox(
                            estimate,
                            partable_lav[:, lav_col][lav_ind[1]];
                            rtol = rtol,
                            atol = atol)
                        push!(correct, is_correct)
                    end

                else

                    lav_ind = findall(
                        (partable_lav.lhs .== String(from)) .&
                        (partable_lav.rhs .== String(to)) .&
                        (partable_lav.op .== type).&
                        (partable_lav.group .== group))

                    if length(lav_ind) == 0
                        throw(ErrorException("Parameter $from $type $to could not be found in the lavaan solution"))
                    elseif length(lav_ind) > 1
                        throw(ErrorException("At least one parameter was found twice in the lavaan solution"))
                    else
                        is_correct = isapprox(
                            estimate,
                            partable_lav[:, lav_col][lav_ind[1]];
                            rtol = rtol,
                            atol = atol)
                        push!(correct, is_correct)
                    end
                end

            end

        end

    end

    return all(correct)
end