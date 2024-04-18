"""
    AIC(sem_fit::SemFit)

Return the akaike information criterion.
"""
AIC(sem_fit::SemFit) = minus2ll(sem_fit) + 2n_par(sem_fit)
