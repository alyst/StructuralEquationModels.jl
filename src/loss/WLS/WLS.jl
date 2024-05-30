##### weighted least squares

############################################################################################
### Types
############################################################################################
"""
Weighted least squares estimation.

# Constructor

    SemWLS(;
        observed::SemObserved, imply::SemImply;
        wls_weight_matrix = nothing,
        wls_weight_matrix_mean = nothing,
        approximate_hessian = false,
        kwargs...)

# Arguments
- `observed`: the `SemObserved` part of the model
- `approximate_hessian::Bool`: should the hessian be swapped for an approximation
- `wls_weight_matrix`: the weight matrix for weighted least squares.
    Defaults to GLS estimation (``0.5*(D^T*kron(S,S)*D)`` where D is the duplication matrix
    and S is the inverse of the observed covariance matrix)
- `wls_weight_matrix_mean`: the weight matrix for the mean part of weighted least squares.
    Defaults to GLS estimation (the inverse of the observed covariance matrix)

# Examples
```julia
my_wls = SemWLS(my_observed, my_imply)
```

# Interfaces
Analytic gradients are available, and for models without a meanstructure, also analytic hessians.

# Extended help
## Implementation
Subtype of `SemLossFunction`.
"""
struct SemWLS{O, I, HE<:HessianEvaluation,Vt,St,C} <: SemLoss{O,I,HE}
    observed::O
    imply::I

    V::Vt
    σₒ::St
    V_μ::C

    SemWLS(observed, imply, ::Type{HE}, args...) where {HE <: HessianEvaluation} =
        new{typeof(observed), typeof(imply), HE, map(typeof, args)...}(observed, imply, args...)
end

############################################################################################
### Constructors
############################################################################################

function SemWLS(observed::SemObserved, imply::SemImply;
                wls_weight_matrix = nothing,
                wls_weight_matrix_mean = nothing,
                approximate_hessian = false)
    # check integrity
    check_observed_vars(observed, imply)

    n_obs = n_man(observed)
    s = vech(obs_cov(observed))
    size(s) == size(imply.Σ) ||
        throw(DimensionMismatch("SemWLS requires implied covariance to be in vech-ed form " *
                                "(vectorized lower triangular part of Σ matrix): $(size(s)) expected, $(size(imply.Σ)) found.\n" *
                                "$(nameof(typeof(imply))) must be constructed with vech=true."))

    # compute V here
    if isnothing(wls_weight_matrix)
        D = duplication_matrix(n_obs)
        S = inv(obs_cov(observed))
        S = kron(S, S)
        wls_weight_matrix = 0.5*(D'*S*D)
    end
    size(wls_weight_matrix) == (length(s), length(s)) ||
        DimensionMismatch("wls_weight_matrix has to be of size $(length(s))×$(length(s))")

    if MeanStructure(imply) == HasMeanStructure
        if isnothing(wls_weight_matrix_mean)
            wls_weight_matrix_mean = inv(obs_cov(observed))
        end
        size(wls_weight_matrix_mean) == (n_obs, n_obs) ||
            DimensionMismatch("wls_weight_matrix_mean has to be of size $(n_obs)×$(n_obs)")
    else
        isnothing(wls_weight_matrix_mean) ||
            @warn "Ignoring wls_weight_matrix_mean since meanstructure is disabled"
        wls_weight_matrix_mean = nothing
    end
    HE = approximate_hessian ? ApproximateHessian : ExactHessian

    return SemWLS(
        observed, imply, HE,
        wls_weight_matrix,
        s,
        wls_weight_matrix_mean,
    )
end

############################################################################
### methods
############################################################################

function evaluate!(objective, gradient, hessian,
                   wls::SemWLS, par
)
    implied = imply(wls)

    if !isnothing(hessian) && (MeanStructure(implied) === HasMeanStructure)
        error("hessian of WLS with meanstructure is not available")
    end

    V = wls.V
    ∇σ = implied.∇Σ

    σ = implied.Σ
    σₒ = wls.σₒ
    σ₋ = σₒ - σ

    isnothing(objective) || (objective = dot(σ₋, V, σ₋))
    if !isnothing(gradient)
        if issparse(∇σ)
            gradient .= (σ₋'*V*∇σ)'
        else # save one allocation
            mul!(gradient, σ₋'*V, ∇σ) # actually transposed, but should be fine for vectors
        end
        gradient .*= -2
    end
    isnothing(hessian) || (mul!(hessian, ∇σ'*V, ∇σ, 2, 0))
    if !isnothing(hessian) && (HessianEvaluation(wls) === ExactHessian)
        ∇²Σ_function! = implied.∇²Σ_function
        ∇²Σ = implied.∇²Σ
        J = -2*(σ₋'*wls.V)'
        ∇²Σ_function!(∇²Σ, J, par)
        hessian .+= ∇²Σ
    end
    if MeanStructure(implied) === HasMeanStructure
        μ = implied.μ
        μₒ = obs_mean(observed(wls))
        μ₋ = μₒ - μ
        V_μ = wls.V_μ
        if !isnothing(objective)
            objective += dot(μ₋, V_μ, μ₋)
        end
        if !isnothing(gradient)
            mul!(gradient, (V_μ*implied.∇μ)', μ₋, -2, 1)
        end
    end

    return objective
end

############################################################################################
### Recommended methods
############################################################################################

update_observed(lossfun::SemWLS, observed::SemObserved; kwargs...) =
    SemWLS(; observed = observed, kwargs...)