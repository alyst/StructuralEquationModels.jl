## connect to Optim.jl as backend

function SemFit(
        optimization_result::Optim.MultivariateOptimizationResults,
        model::AbstractSem,
        start_params)
    trfs = param_transforms(model)
    model_sol = isnothing(trfs) ? optimization_result.minimizer :
        transform_params(trfs, optimization_result.minimizer)
    return SemFit(
        optimization_result.minimum,
        model_sol,
        start_params,
        model,
        optimization_result
    )
end

optimizer(res::Optim.MultivariateOptimizationResults) = Optim.summary(res)
n_iterations(res::Optim.MultivariateOptimizationResults) = Optim.iterations(res)
convergence(res::Optim.MultivariateOptimizationResults) = Optim.converged(res)

function sem_fit(
        optim::SemOptimizerOptim,
        model::AbstractSem,
        start_params::AbstractVector;
        lower_bounds::Union{AbstractVector, AbstractDict, Nothing} = nothing,
        upper_bounds::Union{AbstractVector, AbstractDict, Nothing} = nothing,
        variance_lower_bound::Float64 = 0.0,
        lower_bound = -Inf,
        upper_bound = Inf,
        kwargs...)

    model_trfs = SEM.param_transforms(model)


    # setup lower/upper bounds if the algorithm supports it
    if optim.algorithm isa Optim.Fminbox || optim.algorithm isa Optim.SAMIN
        isnothing(model_trfs) || throw(ArgumentError(
            "Parameter transformations are not supported with bounded " *
            "Optim.jl algorithms (Fminbox or SAMIN)"))

        lbounds = SEM.lower_bounds(lower_bounds, model, default=lower_bound, variance_default=variance_lower_bound)
        ubounds = SEM.upper_bounds(upper_bounds, model, default=upper_bound)
        start_params = clamp.(start_params, lbounds, ubounds)
        result = Optim.optimize(
            Optim.only_fgh!((F, G, H, par) -> evaluate!(F, G, H, model, par)),
            lbounds, ubounds, start_params,
            optim.algorithm,
            optim.options)
    elseif isnothing(model_trfs)
        result = Optim.optimize(
                Optim.only_fgh!((F, G, H, par) -> evaluate!(F, G, H, model, par)),
                start_params,
                optim.algorithm,
                optim.options)
    else
        unconstrained_start_params = inverse_transform_params(model_trfs, start_params)
        result = Optim.optimize(
            Optim.only_fgh!((objective, unconstrained_gradient,
                             unsupported_hessian, unconstrained_vals) ->
                evaluate_unconstrained!(
                    objective, unconstrained_gradient, unsupported_hessian,
                    model, unconstrained_vals)),
            unconstrained_start_params,
            optim.algorithm,
            optim.options,
        )
    end
    return SemFit(result, model, start_params)

end
