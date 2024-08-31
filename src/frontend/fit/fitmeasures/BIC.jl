"""
    BIC(fit::SemFit)

Calculate the *BIC* ([*Bayesian information criterion*](https://en.wikipedia.org/wiki/Bayesian_information_criterion)).
"""
BIC(fit::SemFit) = minus2ll(fit) + log(n_obs(fit))*nparams(fit)