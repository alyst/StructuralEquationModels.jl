"""
    minus2ll(fit::SemFit)

Return the negative 2* log likelihood.
"""
function minus2ll end

############################################################################################
# Single Models
############################################################################################

minus2ll(fit::SemFit) = minus2ll(fit.model, fit)

function minus2ll(term::SemLoss, fit::SemFit)
    minimum = objective(term, fit.solution)
    return minus2ll(term, minimum)
end

minus2ll(term::SemML, minimum::Number) =
    n_obs(term) * (minimum + log(2π) * n_man(term))

# WLS --------------------------------------------------------------------------------------
minus2ll(term::SemWLS, minimum::Number) = missing

# compute likelihood for missing data - H0 -------------------------------------------------
# -2ll = (∑ log(2π)*(nᵢ + mᵢ)) + F*n
function minus2ll(term::SemFIML, minimum::Number)
    obs = observed(term)::SemObservedMissing
    F = minimum * n_obs(obs)
    F += log(2π) * sum(pat -> n_obs(pat)*nobserved_vars(pat), obs.patterns)
    return F
end

# compute likelihood for missing data - H1 -------------------------------------------------
# -2ll =  ∑ log(2π)*(nᵢ + mᵢ) + ln(Σᵢ) + (mᵢ - μᵢ)ᵀ Σᵢ⁻¹ (mᵢ - μᵢ)) + tr(SᵢΣᵢ)
function minus2ll(observed::SemObservedMissing)
    Σ, μ = obs_cov(observed), obs_mean(observed)

    F = 0.0
    for pat in observed.patterns
        nᵢ = n_obs(pat)
        # implied covariance/mean
        Σᵢ = Symmetric(Σ[pat.obs_mask, pat.obs_mask])

        ld = logdet(Σᵢ)
        Σᵢ⁻¹ = LinearAlgebra.inv!(cholesky!(Σᵢ))
        μ_diffᵢ = pat.obs_mean - μ[pat.obs_mask]

        F_pat = ld + dot(μ_diffᵢ, Σᵢ⁻¹, μ_diffᵢ)
        if n_obs(pat) > 1
            F_pat += dot(pat.obs_cov, Σᵢ⁻¹)
        end
        F += (F_pat + log(2π)*nobserved_vars(pat))*n_obs(pat)
    end

    #F *= n_obs(observed)
    return F
end

minus2ll(model::AbstractSem, fit::SemFit) =
    sum(Base.Fix2(minus2ll, fit) ∘ _unwrap ∘ loss, sem_terms(model))
