"""
    df(fit::SemFit)
    df(model::AbstractSem)

Return the degrees of freedom.
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
