"""
    n_obs(fit::SemFit)
    n_obs(model::AbstractSemSingle)
    n_obs(model::SemEnsemble)

Return the number of observed data points.

For ensemble models, return the sum over all submodels.
"""
function n_obs end

n_obs(fit::SemFit) = n_obs(fit.model)
