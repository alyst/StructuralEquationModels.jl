# Ordinary Maximum Likelihood Estimation

############################################################################################
### Types
############################################################################################
"""
Maximum likelihood estimation.

# Constructor

    SemML(observed, imply; approximate_hessian = false)

# Arguments
- `observed::SemObserved`: the observed part of the model
- `imply::SemImply`: [`SemImply`](@ref) instance
- `approximate_hessian::Bool`: if hessian-based optimization is used, should the hessian be swapped for an approximation

# Examples
```julia
my_ml = SemML(my_observed, my_implied)
```

# References

1. Yves Roussel, *"Evaluating the Observed Log-Likelihood Function in
   Two-Level Structural Equation Modeling with Missing Data:
   From Formulas to R Code"*, Psych. 2021, 3(2),
   https://www.mdpi.com/2624-8611/3/2/17

"""
struct SemML{O, I, HE, M} <: SemLoss{O, I, HE}
    observed::O
    imply::I

    obj_offset::Float64 # fixed part of the objective function

    # pre-allocated arrays to store intermediate results in evaluate!()
    obsXobs_1::M
    obsXobs_2::M
    obsXobs_3::M
    varXobs_1::M
    varXvar_1::M
    varXvar_2::M
    varXvar_3::M
end

############################################################################################
### Constructors
############################################################################################

function SemML(observed::SemObserved,
               imply::SemImply;
               minloglikelihood::Bool = false,
               approximate_hessian::Bool = false)
    # check integrity
    check_observed_vars(observed, imply)

    obsXobs = parent(obs_cov(observed))
    nobs = nobserved_vars(imply)
    nvar = nvars(imply)
    obj_offset = minloglikelihood ? -logdet(obs_cov(observed)) - nobs : 0

    return SemML{typeof(observed), typeof(imply),
                 approximate_hessian ? ApproximateHessian : ExactHessian,
                 typeof(obsXobs)}(
        observed, imply, obj_offset,
        similar(obsXobs), similar(obsXobs), similar(obsXobs),
        similar(obsXobs, (nvar, nobs)),
        similar(obsXobs, (nvar, nvar)), similar(obsXobs, (nvar, nvar)),
        similar(obsXobs, (nvar, nvar)))
end

############################################################################################
### objective, gradient, hessian methods
############################################################################################

function evaluate!(
    objective, gradient, hessian,
    ml::SemML,
    par
)
    implied = imply(ml)
    @check_isposdef_Σ(implied, par)

    Σₒ = obs_cov(observed(ml))
    Σ⁻¹ = implied.Σ⁻¹
    # skip Σ⁻¹Σₒ if only the objective is needed
    # since for the objective we only need its trace
    Σ⁻¹Σₒ = !isnothing(gradient) || !isnothing(hessian) ?
        mul!(ml.obsXobs_1, Σ⁻¹, Σₒ) : nothing
    if !isnothing(objective)
        objective = ml.obj_offset + implied.logdet_Σ +
            (!isnothing(Σ⁻¹Σₒ) ? tr(Σ⁻¹Σₒ) : dot(Σ⁻¹, Σₒ))
    end

    if MeanStructure(implied) === HasMeanStructure
        μ₋ = obs_mean(observed(ml)) - implied.μ
        isnothing(objective) || (objective += dot(μ₋, Σ⁻¹, μ₋))
    else
        μ₋ = nothing
    end

    if !isnothing(gradient) || !isnothing(hessian)
        # Σ⁻¹ - Σ⁻¹Σₒ*Σ⁻¹
        Σ⁻¹ = implied.Σ⁻¹
        BLAS.blascopy!(length(Σ⁻¹), parent(Σ⁻¹), 1, ml.obsXobs_2, 1)
        Σ⁻¹mΣ⁻¹ΣₒΣ⁻¹ = mul!(ml.obsXobs_2, Σ⁻¹Σₒ, Σ⁻¹, -1, 1)
        # Σ⁻¹Σₒ (i.e. ml.obsXobs_1) is no longer needed

        evaluate_gradient_hessian!(
            gradient, hessian,
            ml, par, Symmetric(Σ⁻¹mΣ⁻¹ΣₒΣ⁻¹), μ₋
        )
    end
    return objective
end

############################################################################################
### Symbolic Imply Types

function evaluate_gradient_hessian!(
    gradient, hessian,
    ml::SemML{<:Any, <:SemImplySymbolic},
    par, Σ⁻¹mΣ⁻¹ΣₒΣ⁻¹, μ₋
)
    implied = imply(ml)
    if !isnothing(hessian)
        (MeanStructure(implied) === HasMeanStructure) &&
            throw(DomainError(H, "hessian of ML with meanstructure is not supported"))
    end

    # Σ⁻¹ - Σ⁻¹Σₒ*Σ⁻¹
    Σ⁻¹ = implied.Σ⁻¹

    if MeanStructure(implied) === HasMeanStructure
        if !isnothing(gradient)
            ∇Σ = implied.∇Σ
            ∇μ = implied.∇μ
            Σ⁻¹μ₋ = Σ⁻¹*μ₋
            # Σ⁻¹mΣ⁻¹ΣₒΣ⁻¹mμμ = Σ⁻¹*(I - ΣₒΣ⁻¹ - μ₋*μ₋ᵀΣ⁻¹) = Σ⁻¹ - Σ⁻¹Σₒ*Σ⁻¹ - (Σ⁻¹μ₋)*(Σ⁻¹μ₋)ᵀ
            Σ⁻¹mΣ⁻¹ΣₒΣ⁻¹mμμ = copy!(ml.obsXobs_1, Σ⁻¹mΣ⁻¹ΣₒΣ⁻¹)
            X_Xt!(Σ⁻¹mΣ⁻¹ΣₒΣ⁻¹mμμ, Σ⁻¹μ₋, -1, 1)
            mul!(gradient, ∇Σ', vec(Σ⁻¹mΣ⁻¹ΣₒΣ⁻¹mμμ))
            mul!(gradient, ∇μ', Σ⁻¹μ₋, -2, 1)
        end
    elseif !isnothing(gradient) || !isnothing(hessian) # no meanstructure
        ∇Σ = implied.∇Σ
        J = vec(Σ⁻¹mΣ⁻¹ΣₒΣ⁻¹)
        if !isnothing(gradient)
            mul!(gradient, ∇Σ', J)
        end
        if !isnothing(hessian)
            if HessianEvaluation(ml) === ApproximateHessian
                # inner is zero
                Σ_H = Σ⁻¹
                α = 2
                β = 0
            else
                implied.∇²Σ_eval!(hessian, J, par) # inner
                Σ_H = ml.obsXobs_3
                Σ_H .= Σ⁻¹ .- 2 .* Σ⁻¹mΣ⁻¹ΣₒΣ⁻¹ # 2Σ⁻¹Σₒ*Σ⁻¹ - Σ⁻¹
                α = 1
                β = 1
            end
            H_outer = kron(Σ_H, Σ⁻¹)
            Xt_A_X!(hessian, H_outer, ∇Σ, α, β)
        end
    end
    return nothing
end

############################################################################################
### Non-Symbolic Imply Types

function evaluate_gradient_hessian!(
    gradient, hessian,
    ml::SemML,
    par, Σ⁻¹mΣ⁻¹ΣₒΣ⁻¹, μ₋
)
    if !isnothing(hessian)
        error("hessian of ML with non-symbolic imply type is not supported")
    end

    implied = imply(ml)
    Σ⁻¹ = implied.Σ⁻¹

    if !isnothing(gradient)
        S = implied.S
        F⨉I_A⁻¹ = implied.F⨉I_A⁻¹
        I_A⁻¹ = implied.I_A⁻¹

        C = Xt_A_X!(ml.varXvar_1, Σ⁻¹mΣ⁻¹ΣₒΣ⁻¹, F⨉I_A⁻¹, Xt_A_buf = ml.varXobs_1)
        I_A⁻¹⨉S = mul!(ml.varXvar_2, I_A⁻¹, S)
        C⨉S⨉I_Aᵀ⁻¹ = mul!(ml.varXvar_3, C, I_A⁻¹⨉S') # not using Symmetric(C) to allow sparse dispatch

        if !isnothing(μ₋)
            M = implied.M
            k = F⨉I_A⁻¹'*(Σ⁻¹*μ₋)
            # update C⨉S⨉I_Aᵀ⁻¹ and C
            mul!(C⨉S⨉I_Aᵀ⁻¹, k, (I_A⁻¹*(M + S*k))', -1, 1)
            X_Xt!(C, k, -1, 1, check=false)
            mul!(gradient, implied.∇M', k, -2, 0)
        end

        mul!(gradient, implied.∇A', vec(C⨉S⨉I_Aᵀ⁻¹), 2, ifelse(isnothing(μ₋), 0, 1))
        mul!(gradient, implied.∇S', vec(C), 1, 1)
    end

    return nothing
end

############################################################################################
### recommended methods
############################################################################################

update_observed(lossfun::SemML, observed::SemObservedMissing; kwargs...) =
    error("ML estimation does not work with missing data - use FIML instead")

function update_observed(lossfun::SemML, observed::SemObserved; kwargs...)
    if size(lossfun.Σ⁻¹) == size(obs_cov(observed))
        return lossfun
    else
        return SemML(;observed = observed, kwargs...)
    end
end
