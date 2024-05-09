# (Lasso) regularization

############################################################################################
### Types
############################################################################################
"""
Lasso regularization.

# Constructor

    SemLasso(spec::SemSpecification;
             α_lasso, which_lasso, kwargs...)

# Arguments
- `spec`: SEM model specification
- `α_lasso`: hyperparameter for penalty term
- `which_lasso::AbstractVector`: Vector of parameter labels (Symbols)
  or indices that indicate which parameters should be regularized.

# Examples
```julia
my_lasso = SemLasso(spec; α_lasso = 0.02, which_lasso = [:λ₁, :λ₂, :ω₂₃])
```

# Interfaces
Analytic gradients and hessians are available.

# Extended help
## Implementation
Subtype of `SemLossFunction`.
"""
struct SemLasso <: SemLossFunction{ExactHessian}
    α::Float64
    grad_indices::Vector{Int}   # indices of parameters to regularize
    H_diag_indices::Vector{Int} # indices of Hessian diagonal elements to regularize
end

############################################################################
### Constructors
############################################################################

function SemLasso(spec::SemSpecification;
        α_lasso,
        which_lasso,
        kwargs...)

    if eltype(which_lasso) <: Symbol
        param_indices = Dict(param => i for (i, param) in enumerate(params(spec)))
        which_lasso = [param_indices[param] for param in which_lasso]
    end
    H_linind = LinearIndices((nparams(spec), nparams(spec)))
    which_H = [H_linind[i, i] for i in which_lasso]
    return SemLasso(α_lasso, which_lasso, which_H)
end

############################################################################################
### methods
############################################################################################

function evaluate!(
    objective, gradient, hessian,
    lasso::SemLasso,
    imply::SemImply,
    model,
    params)
    obj = NaN
    reg_params = view(params, lasso.grad_indices)
    if !isnothing(objective)
        obj = lasso.α * sum(abs, reg_params)
    end
    if !isnothing(gradient)
        fill!(gradient, 0)
        view(gradient, lasso.grad_indices) .= lasso.α .* sign.(reg_params)
    end
    if !isnothing(hessian)
        fill!(hessian, 0)
    end
    return obj
end

############################################################################################
### Recommended methods
############################################################################################

update_observed(loss::SemLasso, observed::SemObserved; kwargs...) = loss