############################################################################################
### Expectation Maximization Algorithm
############################################################################################

# what about random restarts?

"""
    em_mvn(patterns::AbstractVector{SemObservedMissingPattern};
           start_em = start_em_observed,
           max_iter_em = 100,
           rtol_em = 1e-4,
           kwargs...)

Estimates the covariance matrix and mean vector of the
multivariate normal distribution (MVN)
via expectation maximization (EM) for `observed`.

Returns the tuple of the EM covariance matrix and the EM mean vector.

Based on the EM algorithm for MVN-distributed data with missing values
adapted from the supplementary material to the book *Machine Learning: A Probabilistic Perspective*,
copyright (2010) Kevin Murphy and Matt Dunham: see
[*gaussMissingFitEm.m*](https://github.com/probml/pmtk3/blob/master/toolbox/BasicModels/gauss/sub/gaussMissingFitEm.m) and
[*emAlgo.m*](https://github.com/probml/pmtk3/blob/master/toolbox/Algorithms/optimization/emAlgo.m) scripts.
"""
function em_mvn(
    patterns::AbstractVector{<:SemObservedMissingPattern};
    start_em = start_em_observed,
    max_iter_em::Integer = 100,
    rtol_em::Number = 1e-4,
    max_nobs_em::Union{Integer, Nothing} = nothing,
    verbose::Bool = false,
    min_eigval::Union{Number, Nothing} = nothing,
    kwargs...)

    n_man = SEM.n_man(patterns[1])

    verbose && @info "Estimating N(μ, Σ) for complete observations..."
    𝔼x_full = zeros(n_man)
    𝔼xxᵀ_full = zeros(n_man, n_man)
    nobs_full = 0
    for pat in patterns
        if nmissed_vars(pat) == 0
            𝔼x_full .+= sum(pat.data, dims=2)
            mul!(𝔼xxᵀ_full, pat.data, pat.data', 1, 1)
            nobs_full += n_obs(pat)
        end
    end
    if nobs_full == 0
        @warn "No full cases in data"
    end

    verbose && @info "Estimating initial μ and Σ..."
    Σ₀, μ = start_em(patterns; kwargs...)
    Σ = convert(Matrix, Σ₀)
    @assert all(isfinite, Σ) all(isfinite, μ)
    Σ_prev, μ_prev = copy(Σ), copy(μ)

    iter = 0
    converged = false
    Δμ_rel = NaN
    ΔΣ_rel = NaN
    progress = Progress(max_iter_em, dt=1.0, showspeed=true, desc="EM inference of MVN(μ, Σ)")
    while !converged && (iter < max_iter_em)
        em_step!(Σ, μ, Σ_prev, μ_prev, patterns,
                 𝔼xxᵀ_full, 𝔼x_full, nobs_full; max_nobs_em, min_eigval)

        if iter > 0
            Δμ = norm(μ - μ_prev)
            ΔΣ = norm(Σ - Σ_prev)
            Δμ_rel = Δμ / max(norm(μ_prev), norm(μ))
            ΔΣ_rel = ΔΣ / max(norm(Σ_prev), norm(Σ))
            #@info "Iteration #$iter: ΔΣ=$(ΔΣ) ΔΣ/Σ=$(ΔΣ_rel) Δμ=$(Δμ) Δμ/μ=$(Δμ_rel)"
            # converged = isapprox(ll, ll_prev; rtol = rtol)
            converged = ΔΣ_rel <= rtol_em && Δμ_rel <= rtol_em
        end
        if !converged
            Σ, Σ_prev = Σ_prev, Σ
            μ, μ_prev = μ_prev, μ
        end
        iter += 1
        next!(progress, step=1, showvalues=[("ΔΣ/Σ", ΔΣ_rel), ("Δμ/μ", Δμ_rel)])
    end
    finish!(progress)

    if !converged
        @warn "EM inference for MVN missing data did not converge in $iter iterations.\n" *
              "Final tolerances: ΔΣ/Σ=$(ΔΣ_rel), Δμ/μ=$(Δμ_rel).\n" *
              "Likelihood for FIML is not interpretable.\n" *
              "Maybe try passing different starting values via 'start_em = ...' "
    else
        verbose && @info "EM for MVN missing data converged in $iter iterations: ΔΣ/Σ=$(ΔΣ_rel), Δμ/μ=$(Δμ_rel)."
    end

    return Σ, μ
end

# E and M steps -----------------------------------------------------------------------------

function em_step!(Σ::AbstractMatrix, μ::AbstractVector,
                  Σ₀::AbstractMatrix, μ₀::AbstractVector,
                  patterns::AbstractVector{<:SemObservedMissingPattern},
                  𝔼xxᵀ_full::AbstractMatrix, 𝔼x_full::AbstractVector, nobs_full::Integer;
                  max_nobs_em::Union{Integer, Nothing} = nothing,
                  min_eigval::Union{Number, Nothing} = nothing
)
    # E step, update 𝔼x and 𝔼xxᵀ
    copy!(μ, 𝔼x_full)
    copy!(Σ, 𝔼xxᵀ_full)
    nobs_used = nobs_full
    mul!(Σ, μ₀, μ₀', -nobs_used, 1)
    axpy!(-nobs_used, μ₀, μ)

    # Compute the expected sufficient statistics
    for pat in patterns
        (nmissed_vars(pat) == 0) && continue # full cases already accounted for

        # observed and unobserved vars
        u = pat.miss_mask
        o = pat.obs_mask

        # compute cholesky to speed-up ldiv!()
        Σ₀oo_chol = cholesky(Symmetric(Σ₀[o, o]))
        Σ₀uo = Σ₀[u, o]
        μ₀u = μ₀[u]
        μ₀o = μ₀[o]

        # get pattern observations
        nobs = !isnothing(max_nobs_em) ? min(max_nobs_em, n_obs(pat)) : n_obs(pat)
        zo = nobs < n_obs(pat) ?
            pat.data[:, sort!(sample(1:n_obs(pat), nobs, replace = false))] :
            copy(pat.data)
        zo .-= μ₀o # subtract current mean from observations

        𝔼zo = sum(zo, dims = 2)
        𝔼zu = fill!(similar(μ₀u), 0)

        𝔼zzᵀuo = fill!(similar(Σ₀uo), 0)
        𝔼zzᵀuu = nobs * Σ₀[u, u]
        mul!(𝔼zzᵀuu, Σ₀uo, Σ₀oo_chol \ Σ₀uo', -nobs, 1)

        # loop through observations
        yᵢo = similar(μ₀o)
        𝔼zᵢu = similar(μ₀u)
        @inbounds for zᵢo in eachcol(zo)
            ldiv!(yᵢo, Σ₀oo_chol, zᵢo)
            mul!(𝔼zᵢu, Σ₀uo, yᵢo)
            mul!(𝔼zzᵀuu, 𝔼zᵢu, 𝔼zᵢu', 1, 1)
            mul!(𝔼zzᵀuo, 𝔼zᵢu, zᵢo', 1, 1)
            𝔼zu .+= 𝔼zᵢu
        end
        # correct 𝔼zzᵀ by adding back μ₀×𝔼z' + 𝔼z'×μ₀
        mul!(𝔼zzᵀuo, μ₀u, 𝔼zo', 1, 1)
        mul!(𝔼zzᵀuo, 𝔼zu, μ₀o', 1, 1)

        mul!(𝔼zzᵀuu, μ₀u, 𝔼zu', 1, 1)
        mul!(𝔼zzᵀuu, 𝔼zu, μ₀u', 1, 1)

        𝔼zzᵀoo = zo * zo'
        mul!(𝔼zzᵀoo, μ₀o, 𝔼zo', 1, 1)
        mul!(𝔼zzᵀoo, 𝔼zo, μ₀o', 1, 1)

        # update Σ and μ
        Σ[o,o] .+= 𝔼zzᵀoo
        Σ[u,o] .+= 𝔼zzᵀuo
        Σ[o,u] .+= 𝔼zzᵀuo'
        Σ[u,u] .+= 𝔼zzᵀuu

        μ[o] .+= 𝔼zo
        μ[u] .+= 𝔼zu

        nobs_used += nobs
    end

    # M step, update em_model
    lmul!(1/nobs_used, Σ)
    lmul!(1/nobs_used, μ)
    # at this point μ = μ - μ₀
    # and Σ = Σ + (μ - μ₀)×(μ - μ₀)' - μ₀×μ₀'
    mul!(Σ, μ, μ₀', -1, 1)
    mul!(Σ, μ₀, μ', -1, 1)
    mul!(Σ, μ, μ', -1, 1)
    μ .+= μ₀

    isnothing(min_eigval) || copyto!(Σ, trunc_eigvals(Σ, min_eigval))
    # ridge Σ
    # while !isposdef(Σ)
    #     Σ += 0.5I
    # end

    # diagonalization
    #if !isposdef(Σ)
    #    print("Matrix not positive definite")
    #    Σ .= 0
    #    Σ[diagind(em_model.Σ)] .= diag(Σ)
    #else
        # Σ = Σ
    #end

    return Σ, μ
end

# generate starting values -----------------------------------------------------------------

# use μ and Σ of full cases
function start_em_observed(patterns::AbstractVector{<:SemObservedMissingPattern}; kwargs...)

    fullpat = patterns[1]
    if (nmissed_vars(fullpat) == 0) && (n_obs(fullpat) > 1)
        μ = copy(fullpat.obs_mean)
        Σ = copy(parent(fullpat.obs_cov))
        if !isposdef(Σ)
            Σ = Diagonal(Σ)
        end
        return Σ, μ
    else
        return start_em_simple(observed, kwargs...)
    end

end

# use μ = O and Σ = I
function start_em_simple(patterns::AbstractVector{<:SemObservedMissingPattern}; kwargs...)
    nvars = n_man(first(patterns))
    μ = zeros(nvars)
    Σ = rand(nvars, nvars)
    Σ = Σ*Σ'
    # Σ = Matrix(1.0I, n_man, n_man)
    return Σ, μ
end

# set to passed values
function start_em_set(patterns::AbstractVector{<:SemObservedMissingPattern};
                      obs_cov::AbstractMatrix, obs_mean::AbstractVector, kwargs...)
    return copy(obs_cov), copy(obs_mean)
end