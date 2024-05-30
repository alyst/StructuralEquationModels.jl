# (Ridge) regularization

############################################################################################
### Types
############################################################################################
"""
Ridge regularization.

# Constructor

    SemRidge(spec::SemSpecification; params)

# Arguments
- `spec`: SEM model specification
- `params::Vector`: Vector of parameter IDs (Symbols)
  or indices that indicate which parameters should be regularized.

# Examples
```julia
my_ridge = SemRidge(spec; params = [:λ₁, :λ₂, :ω₂₃])
```

# Interfaces
Analytic gradients and hessians are available.

# Extended help
## Implementation
Subtype of `SemLossFunction`.
"""
struct SemRidge <: AbstractLoss{ExactHessian}
    param_inds::Vector{Int}   # indices of parameters to regularize
    H_diag_inds::Vector{Int} # indices of Hessian diagonal elements to regularize
end

############################################################################
### Constructors
############################################################################

function SemRidge(
    spec::SemSpecification,
    params::AbstractVector
)
    param_inds = eltype(params) <: Symbol ? param_indices(spec, params) : params
    H_linind = LinearIndices((nparams(spec), nparams(spec)))
    return SemRidge(param_inds, [H_linind[i, i] for i in param_inds])
end

############################################################################################
### methods
############################################################################################

function evaluate!(
    objective, gradient, hessian,
    ridge::SemRidge,
    params
)
    obj = NaN
    reg_params = view(params, ridge.param_inds)
    if !isnothing(objective)
        obj = sum(abs2, reg_params)
    end
    if !isnothing(gradient)
        fill!(gradient, 0)
        view(gradient, ridge.param_inds) .= 2 .* reg_params
    end
    if !isnothing(hessian)
        fill!(hessian, 0)
        view(hessian, ridge.H_diag_inds) .= 2
    end
    return obj
end

############################################################################################
### Recommended methods
############################################################################################

update_observed(loss::SemRidge, observed::SemObserved; kwargs...) = loss