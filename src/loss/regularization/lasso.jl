# (Lasso) regularization

############################################################################################
### Types
############################################################################################
"""
Lasso regularization.

# Constructor

    SemLasso(spec::SemSpecification; params)

# Arguments
- `spec`: SEM model specification
- `params::AbstractVector`: Vector of parameter IDs (Symbols)
  or indices that indicate which parameters should be regularized.

# Examples
```julia
my_lasso = SemLasso(spec; params = [:λ₁, :λ₂, :ω₂₃])
```

# Interfaces
Analytic gradients and hessians are available.

# Extended help
## Implementation
Subtype of `SemLossFunction`.
"""
struct SemLasso <: AbstractLoss{ExactHessian}
    param_inds::Vector{Int}   # indices of parameters to regularize
end

function SemLasso(
    spec::SemSpecification,
    params::AbstractVector,
)
    param_inds = eltype(params) <: Symbol ? param_indices(spec, params) : params
    return SemLasso(param_inds)
end

function evaluate!(
    objective, gradient, hessian,
    lasso::SemLasso,
    params
)
    obj = NaN
    reg_params = view(params, lasso.param_inds)
    if !isnothing(objective)
        obj = sum(abs, reg_params)
    end
    if !isnothing(gradient)
        fill!(gradient, 0)
        view(gradient, lasso.param_inds) .= sign.(reg_params)
    end
    if !isnothing(hessian)
        fill!(hessian, 0)
    end
    return obj
end
