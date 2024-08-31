"""
    df(fit::SemFit)
    df(model::AbstractSem)

Get the *degrees of freedom* for the SEM model.

The degrees of freedom for the SEM model with *N* observed variables
is the difference between the number of parameters
required to define the *N×N* covariance matrix (*½N(N+1)*)
(plus *N* parameters for the observed means vector, if present),
and the number of model parameters, [`nparams(model)`](@ref nparams).
"""
function df end

df(fit::SemFit) = df(fit.model)

df(model::AbstractSem) = n_dp(model) - nparams(model)

# length of Σ and μ (if present)
function n_dp(imply::SemImply)
    nman = nobserved_vars(imply)
    ndp = 0.5(nman^2 + nman)
    if !isnothing(imply.μ)
        ndp += nman
    end
    return ndp
end

n_dp(term::SemLoss) = n_dp(imply(term))

n_dp(model::AbstractSem) = sum(n_dp∘loss, sem_terms(model))
