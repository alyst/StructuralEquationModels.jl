# (Ridge) regularization

############################################################################################
### Types
############################################################################################
"""
Ridge regularization.

# Constructor

    SemRidge(spec::SemSpecification;
             α_ridge, which_ridge, kwargs...)

# Arguments
- `spec`: SEM model specification
- `α_ridge`: hyperparameter for penalty term
- `which_ridge::Vector`: Vector of parameter labels (Symbols)
  or indices that indicate which parameters should be regularized.

# Examples
```julia
my_ridge = SemRidge(spec; α_ridge = 0.02, which_ridge = [:λ₁, :λ₂, :ω₂₃])
```

# Interfaces
Analytic gradients and hessians are available.

# Extended help
## Implementation
Subtype of `SemLossFunction`.
"""
struct SemRidge <: SemLossFunction{ExactHessian}
    α::Float64
    grad_indices::Vector{Int}   # indices of parameters to regularize
    H_diag_indices::Vector{Int} # indices of Hessian diagonal elements to regularize
end

############################################################################
### Constructors
############################################################################

function SemRidge(spec::SemSpecification;
        α_ridge,
        which_ridge,
        kwargs...)

    if eltype(which_ridge) <: Symbol
        param_indices = Dict(param => i for (i, param) in enumerate(params(spec)))
        which_ridge = [param_indices[param] for param in which_ridge]
    end
    H_linind = LinearIndices((nparams(spec), nparams(spec)))
    which_H = [H_linind[i, i] for i in which_ridge]
    return SemRidge(α_ridge, which_ridge, which_H)
end

############################################################################################
### methods
############################################################################################

function evaluate!(
    objective, gradient, hessian,
    ridge::SemRidge,
    imply::SemImply,
    model,
    params)
    obj = NaN
    reg_params = view(params, ridge.grad_indices)
    if !isnothing(objective)
        obj = ridge.α * sum(abs2, reg_params)
    end
    if !isnothing(gradient)
        fill!(gradient, 0)
        view(gradient, ridge.grad_indices) .= (2*ridge.α) .* reg_params
    end
    if !isnothing(hessian)
        fill!(hessian, 0)
        view(hessian, ridge.H_diag_indices) .= (2*ridge.α)
    end
    return obj
end

############################################################################################
### Recommended methods
############################################################################################

update_observed(loss::SemRidge, observed::SemObserved; kwargs...) = loss