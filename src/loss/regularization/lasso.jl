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
    param_inds::Vector{Int}   # indices of parameters to regularize
end

function SemLasso(spec::SemSpecification;
    α::Number,
    params::AbstractVector,
)
    param_inds = eltype(params) <: Symbol ? param_indices(spec, params) : params
    return SemLasso(α, param_inds)
end

function evaluate!(
    objective, gradient, hessian,
    lasso::SemLasso,
    imply::SemImply,
    model,
    params
)
    obj = NaN
    reg_params = view(params, lasso.param_inds)
    if !isnothing(objective)
        obj = lasso.α * sum(abs, reg_params)
    end
    if !isnothing(gradient)
        fill!(gradient, 0)
        view(gradient, lasso.param_inds) .= lasso.α .* sign.(reg_params)
    end
    if !isnothing(hessian)
        fill!(hessian, 0)
    end
    return obj
end
