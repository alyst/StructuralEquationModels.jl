"""
    n_man(fit::SemFit)
    n_man(model::AbstractSemSingle)

Return the number of manifest variables.
"""
function n_man end

n_man(fit::SemFit) = n_man(fit.model)
