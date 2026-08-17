############################################################################################
### get number of parameters
############################################################################################
"""
    nparams(sem_fit::SemFit; model::Bool = true)
    nparams(model::AbstractSem; model::Bool = true)

Return the number of parameters. With `model = true` (default), return the number
of model-space parameters. With `model = false`, return the number of
unconstrained (optimizer-space) parameters -- see [`nparams_unconstrained`](@ref).
"""
nparams(obj::Any; model::Bool = true) = model ? length(params(obj)) : nparams_unconstrained(obj)

"""
    nparams_unconstrained(obj)

Return the number of unconstrained (optimizer-space) parameters of `obj`.
Equals to `nparams(obj)` unless `obj` contains non-identity [`ParamTransforms`](@ref)
that define some model-space parameters (e.g. derived parameters defined as linear
combinations of others), which then don't have independent unconstrained
coordinates.
"""
function nparams_unconstrained(model::AbstractSem)
    trfs = param_transforms(model)
    return isnothing(trfs) ? nparams(model) : nparams_unconstrained(trfs)
end

# fallback method
nparams_unconstrained(obj::Any) = nparams(obj)
