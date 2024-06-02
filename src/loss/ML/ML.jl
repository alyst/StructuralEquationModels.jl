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

    obj_base::Float64 # fixed part of the objective function

    # pre-allocated arrays to store intermediate results in evaluate!()
    obsXobs_1::M
    obsXobs_2::M
    obsXobs_3::M
    obsXvar_1::M
    varXvar_1::M
    varXvar_2::M
    varXvar_3::M
end

############################################################################################
### Constructors
############################################################################################

function SemML(observed::SemObserved,
               imply::SemImply;
               approximate_hessian::Bool = false)
    # check integrity
    check_observed_vars(observed, imply)

    obsXobs = parent(obs_cov(observed))
    nobs = nobserved_vars(imply)
    nvar = nvars(imply)
    obj_base = -logdet(obs_cov(observed)) - nobs

    return SemML{typeof(observed), typeof(imply),
                 approximate_hessian ? ApproximateHessian : ExactHessian,
                 typeof(obsXobs)}(
        observed, imply, obj_base,
        similar(obsXobs), similar(obsXobs), similar(obsXobs),
        similar(obsXobs, (nobs, nvar)),
        similar(obsXobs, (nvar, nvar)), similar(obsXobs, (nvar, nvar)),
        similar(obsXobs, (nvar, nvar)))
end

############################################################################################
### objective, gradient, hessian methods
############################################################################################

############################################################################################
### Symbolic Imply Types

function evaluate!(
    objective, gradient, hessian,
    ml::SemML{<:Any, <:SemImplySymbolic},
    par
)
    implied = imply(ml)

    if !isnothing(hessian)
        (MeanStructure(implied) === HasMeanStructure) &&
            throw(DomainError(H, "hessian of ML + meanstructure is not available"))
    end

    Σ = implied.Σ
    Σₒ = obs_cov(observed(ml))

    Σ⁻¹ = copy!(ml.obsXobs_1, Σ)
    Σ_chol = cholesky!(Symmetric(Σ⁻¹); check = false)
    if !isposdef(Σ_chol)
        #@warn "∑⁻¹ is not positive definite"
        isnothing(objective) || (objective = non_posdef_return(par))
        isnothing(gradient) || fill!(gradient, 1)
        isnothing(hessian) || copyto!(hessian, I)
        return objective
    end
    logdet_Σ = logdet(Σ_chol)
    Σ⁻¹ = LinearAlgebra.inv!(Σ_chol)
    Σ⁻¹Σₒ = mul!(ml.obsXobs_2, Σ⁻¹, Σₒ)
    isnothing(objective) || (objective = ml.obj_base + logdet_Σ + tr(Σ⁻¹Σₒ))

    if MeanStructure(implied) === HasMeanStructure
        μ = implied.μ
        μₒ = obs_mean(observed(ml))
        μ₋ = μₒ - μ

        isnothing(objective) || (objective += dot(μ₋, Σ⁻¹, μ₋))
        if !isnothing(gradient)
            ∇Σ = implied.∇Σ
            ∇μ = implied.∇μ
            μ₋ᵀΣ⁻¹ = μ₋'*Σ⁻¹
            mul!(gradient, ∇Σ', vec(Σ⁻¹*(I - mul!(ml.obsXobs_3, Σₒ, Σ⁻¹) - μ₋*μ₋ᵀΣ⁻¹)))
            mul!(gradient, ∇μ', μ₋ᵀΣ⁻¹', -2, 1)
        end
    elseif !isnothing(gradient) || !isnothing(hessian)
        ∇Σ = implied.∇Σ
        Σ⁻¹ΣₒΣ⁻¹ = mul!(ml.obsXobs_3, Σ⁻¹Σₒ, Σ⁻¹)
        J = vec(Σ⁻¹ - Σ⁻¹ΣₒΣ⁻¹)'
        if !isnothing(gradient)
            mul!(gradient, ∇Σ', J')
        end
        if !isnothing(hessian)
            if HessianEvaluation(ml) === ApproximateHessian
                mul!(hessian, ∇Σ'*kron(Σ⁻¹, Σ⁻¹), ∇Σ, 2, 0)
            else
                ∇²Σ_function! = implied.∇²Σ_function
                ∇²Σ = implied.∇²Σ
                # inner
                ∇²Σ_function!(∇²Σ, J, par)
                # outer
                H_outer = kron(2Σ⁻¹ΣₒΣ⁻¹ - Σ⁻¹, Σ⁻¹)
                mul!(hessian, ∇Σ'*H_outer, ∇Σ)
                hessian .+= ∇²Σ
            end
        end
    end
    return objective
end

############################################################################################
### Non-Symbolic Imply Types

function evaluate!(
    objective, gradient, hessian,
    ml::SemML,
    par
)
    if !isnothing(hessian)
        error("hessian of ML + non-symbolic imply type is not available")
    end

    implied = imply(ml)

    Σ = implied.Σ
    Σₒ = obs_cov(observed(ml))

    Σ⁻¹ = copy!(ml.obsXobs_1, Σ)
    Σ_chol = cholesky!(Symmetric(Σ⁻¹); check = false)
    if !isposdef(Σ_chol)
        #@warn "Σ⁻¹ is not positive definite"
        isnothing(objective) || (objective = non_posdef_return(par))
        isnothing(gradient) || fill!(gradient, 1)
        isnothing(hessian) || copyto!(hessian, I)
        return objective
    end
    logdet_Σ = logdet(Σ_chol)
    Σ⁻¹ = LinearAlgebra.inv!(Σ_chol)
    Σ⁻¹Σₒ = mul!(ml.obsXobs_2, Σ⁻¹, Σₒ)

    if !isnothing(objective)
        objective = ml.obj_base + logdet_Σ + tr(Σ⁻¹Σₒ)

        if MeanStructure(implied) === HasMeanStructure
            μ = implied.μ
            μₒ = obs_mean(observed(ml))
            μ₋ = μₒ - μ
            objective += dot(μ₋, Σ⁻¹, μ₋)
        end
    end

    if !isnothing(gradient)
        S = implied.S
        F⨉I_A⁻¹ = implied.F⨉I_A⁻¹
        I_A⁻¹ = implied.I_A⁻¹
        ∇A = implied.∇A
        ∇S = implied.∇S

        # reuse Σ⁻¹Σₒ to calculate I-Σ⁻¹Σₒ
        one_Σ⁻¹Σₒ = Σ⁻¹Σₒ
        lmul!(-1, one_Σ⁻¹Σₒ)
        one_Σ⁻¹Σₒ[diagind(one_Σ⁻¹Σₒ)] .+= 1

        C = mul!(ml.varXvar_1, F⨉I_A⁻¹',
                 mul!(ml.obsXvar_1,
                      Symmetric(mul!(ml.obsXobs_3, one_Σ⁻¹Σₒ, Σ⁻¹)), F⨉I_A⁻¹))
        mul!(gradient, ∇A',
             vec(mul!(ml.varXvar_3,
                    Symmetric(C),
                    mul!(ml.varXvar_2, S, I_A⁻¹'))), 2, 0)
        mul!(gradient, ∇S', vec(C), 1, 1)

        if MeanStructure(implied) === HasMeanStructure
            μ = implied.μ
            μₒ = obs_mean(observed(ml))
            ∇M = implied.∇M
            M = implied.M
            μ₋ = μₒ - μ
            μ₋ᵀΣ⁻¹ = μ₋'*Σ⁻¹
            k = μ₋ᵀΣ⁻¹*F⨉I_A⁻¹
            mul!(gradient, ∇M', k', -2, 1)
            mul!(gradient, ∇A', vec(mul!(ml.varXvar_1, k', (I_A⁻¹*(M + S*k'))')), -2, 1)
            mul!(gradient, ∇S', vec(mul!(ml.varXvar_2, k', k)), -1, 1)
        end
    end

    return objective
end

############################################################################################
### additional functions
############################################################################################

function non_posdef_return(par)
    if eltype(par) <: AbstractFloat
        return floatmax(eltype(par))
    else
        return typemax(eltype(par))
    end
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
