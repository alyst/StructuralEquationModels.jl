fit_measures(fit) =
    fit_measures(
        fit,
        nparams,
        df,
        AIC,
        BIC,
        RMSEA,
        χ²,
        p_value,
        minus2ll
        )

fit_measures(fit, args...) =
    Dict(Symbol(arg) => arg(fit) for arg in args)

"""
    fit_measures(sem_fit, args...)

Return a default set of fit measures or the fit measures passed as `arg...`.
"""
function fit_measures end