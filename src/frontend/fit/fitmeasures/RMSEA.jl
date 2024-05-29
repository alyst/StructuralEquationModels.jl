"""
    RMSEA(fit::SemFit)

Return the RMSEA.
"""
function RMSEA end

RMSEA(fit::SemFit) = RMSEA(fit, fit.model)

RMSEA(fit::SemFit, model::AbstractSemSingle) =
    RMSEA(df(fit), χ²(fit), n_obs(fit))

RMSEA(fit::SemFit, model::SemEnsemble) =
    sqrt(length(model.sems))*RMSEA(df(fit), χ²(fit), n_obs(fit))

function RMSEA(df, chi2, n_obs)
    rmsea = (chi2 - df) / (n_obs*df)
    rmsea > 0 ? nothing : rmsea = 0
    return sqrt(rmsea)
end