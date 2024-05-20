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
    param_inds::Vector{Int}   # indices of parameters to regularize
    H_diag_inds::Vector{Int} # indices of Hessian diagonal elements to regularize
end

############################################################################
### Constructors
############################################################################

function SemRidge(spec::SemSpecification;
    α_ridge::Number,
    which_ridge::AbstractVector,
    kwargs...
)
    param_inds = eltype(params) <: Symbol ? param_indices(spec, which_ridge) : which_ridge
    H_linind = LinearIndices((nparams(spec), nparams(spec)))
    return SemRidge(α_ridge, param_inds, [H_linind[i, i] for i in param_inds])
end

############################################################################################
### methods
############################################################################################

function evaluate!(
    objective, gradient, hessian,
    ridge::SemRidge,
    imply::SemImply,
    model,
    params
)
    obj = NaN
    reg_params = view(params, ridge.param_inds)
    if !isnothing(objective)
        obj = ridge.α * sum(abs2, reg_params)
    end
    if !isnothing(gradient)
        fill!(gradient, 0)
        view(gradient, ridge.param_inds) .= (2*ridge.α) .* reg_params
    end
    if !isnothing(hessian)
        fill!(hessian, 0)
        view(hessian, ridge.H_diag_inds) .= (2*ridge.α)
    end
    return obj
end

############################################################################################
### Recommended methods
############################################################################################

update_observed(loss::SemRidge, observed::SemObserved; kwargs...) = loss