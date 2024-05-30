"""
    RMSEA(fit::SemFit)

Return the RMSEA.
"""
RMSEA(fit::SemFit) = RMSEA(fit, fit.model)

RMSEA(fit::SemFit, model::AbstractSem) =
    sqrt(nsem_terms(model)) * RMSEA(df(fit), χ²(fit), n_obs(fit))

RMSEA(df::Number, chi2::Number, n_obs::Number) =
    sqrt(max((chi2 - df) / (n_obs * df), 0.0))
