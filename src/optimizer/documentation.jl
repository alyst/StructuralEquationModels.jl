"""
    sem_fit(optim::SemOptimizer, model::AbstractSem; start_params = nothing, kwargs...)

Return the fitted `model`.

# Arguments
- `optim`: [`SemOptimizer`](@ref) to use for fitting
- `model`: `AbstractSem` to fit
- `start_params`: starting parameter values as a vector or dictionary, or a
                  function that computes them (1)
- `kwargs...`: keyword arguments, passed to starting value functions

(1) available functions are `start_fabin3`, `start_simple` and `start_partable`.
For more information, we refer to the individual documentations and the online documentation on [Starting values](@ref).

# Examples
```julia
sem_fit(
    my_model;
    start_params = start_simple,
    start_covariances_latent = 0.5)
```
"""
function sem_fit(optim::SemOptimizer, model::AbstractSem;
                 start_params = nothing, kwargs...)
    prep_start_params = prepare_start_params(start_params, model; kwargs...)
    sem_fit(optim, model, prep_start_params; kwargs...)
end

sem_fit(model::AbstractSem; engine::Symbol=:Optim, start_params = nothing, kwargs...) =
    sem_fit(SemOptimizer(; engine, kwargs...), model; start_params, kwargs...)

# fallback method
sem_fit(optim::SemOptimizer, model::AbstractSem, start_params; kwargs...) =
    error("Optimizer $(optim) support not implemented.")

"""
    evaluate_unconstrained!(
        objective,
        unconstrained_gradient,
        unsupported_hessian,
        model,
        unconstrained_vals,
    )

Evaluate `model` at `unconstrained_vals` using its declared parameter
transformations.

`unconstrained_vals` is transformed to model values before evaluation. If
`unconstrained_gradient` is not `nothing`, the model gradient is pulled back to
unconstrained space and written into it. Temporary vectors are local to the
call, so concurrent evaluations do not share mutable workspace.

Pass `nothing` for `unconstrained_gradient` when only the objective is needed.
Hessian evaluation with non-identity parameter transforms is not currently
supported, so `unsupported_hessian` must be `nothing`.
"""
function evaluate_unconstrained!(
    objective,
    unconstrained_gradient,
    unsupported_hessian,
    model::AbstractSem,
    unconstrained_vals::AbstractVector,
)

    param_trfs = param_transforms(model)
    isnothing(param_trfs) || isnothing(unsupported_hessian) ||
        throw(ArgumentError(
            "Hessian evaluation is not yet supported with non-identity " *
            "parameter transformations"))

            # calculate scalar derivatives if unconstrained_gradient is requested
    scalar_derivs = isnothing(unconstrained_gradient) || isnothing(param_trfs) ? nothing :
        similar(unconstrained_vals)
    model_vals = isnothing(param_trfs) ? unconstrained_vals :
        transform_params!(similar(unconstrained_vals), scalar_derivs, param_trfs, unconstrained_vals)

    model_grad = isnothing(scalar_derivs) ? unconstrained_gradient : similar(unconstrained_vals)
    res = evaluate!(objective, model_grad, nothing, model, model_vals)
    isnothing(scalar_derivs) || pullback_param_gradient!(
        unconstrained_gradient, model_grad, model_vals,
        scalar_derivs, param_trfs)
    return res
end

function prepare_start_params(start_params, model::AbstractSem;
                              start_params_jitter::Number = 0,
                              start_params_shrink::Real = sqrt(eps(Float64)),
                              kwargs...)
    if isnothing(start_params)
        # default function for starting parameters
        # FABIN3 for single models, simple algorithm for ensembles
        sems = sem_terms(model)
        start_params = length(sems) == 1 ?
            start_fabin3(loss(sems[1]); kwargs...) :
            start_simple(model; kwargs...)
    end
    if start_params isa AbstractVector
        (length(start_params) == nparams(model)) ||
            throw(DimensionMismatch("The length of `start_params` vector ($(length(start_params))) does not match the number of model parameters ($(nparams(model)))."))
    elseif start_params isa AbstractDict
        start_params = [start_params[param] for param in params(model)]
    else # function
        start_params = start_params(model; kwargs...)
    end
    @assert start_params isa AbstractVector{<:Number}
    @assert length(start_params) == nparams(model)
    if start_params_jitter != 0
        transforms = param_transforms(model)
        if isnothing(transforms)
            start_params .+= randn(length(start_params)) * start_params_jitter
        else
            unconstrained_start_params = inverse_transform_params(
                transforms, start_params)
            unconstrained_start_params .+=
                randn(length(start_params)) * start_params_jitter
            start_params = transform_params(transforms, unconstrained_start_params)
            if start_params_shrink > 0
                start_params = project_to_interior(
                    start_params, transforms;
                    shrink = start_params_shrink,
                )
            end
        end
    end
    return start_params
end

# define a vector of parameter lower bounds: use user-specified vector as is
function lower_bounds(bounds::AbstractVector, model::AbstractSem;
                      default::Number, variance_default::Number)
    length(bound) == nparams(model) ||
        throw(DimensionMismatch("The length of `bounds` vector ($(length(bounds))) does not match the number of model parameters ($(nparams(model)))."))
    return bounds
end

# define a vector of parameter lower bounds given a dictionary and default values
function lower_bounds(bounds::Union{AbstractDict, Nothing}, model::AbstractSem;
                      default::Number, variance_default::Number)
    varparams = Set{Symbol}()
    for term in loss_terms(model)
        if issemloss(term)
            push!.(Ref(varparams), variance_params(imply(term).ram_matrices))
        end
    end
    res = [begin
        def = in(p, varparams) ? variance_default : default
        isnothing(bounds) ? def : get(bounds, p, def)
    end for p in SEM.params(model)]

    return res
end

# define a vector of parameter upper bounds: use user-specified vector as is
function upper_bounds(bounds::AbstractVector, model::AbstractSem;
                      default::Number)
    length(bound) == nparams(model) ||
        throw(DimensionMismatch("The length of `bounds` vector ($(length(bounds))) does not match the number of model parameters ($(nparams(model)))."))
    return bounds
end

# define a vector of parameter lower bounds given a dictionary and default values
function upper_bounds(bounds::Union{AbstractDict, Nothing}, model::AbstractSem;
                      default::Number)
    res = [begin
        def = default
        isnothing(bounds) ? def : get(bounds, p, def)
    end for p in SEM.params(model)]

    return res
end
