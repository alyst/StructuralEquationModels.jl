"""
    SemHinge{B} <: SemLossFunction{ExactHessian}

Hinge regularization.

Implements *hinge* a.k.a *rectified linear unit* (*ReLU*) loss function:
```math
f_{t}(x) = \\begin{cases} 0 & \\text{if}\\ x \\leq t \\\\
                        (x - t) & \\text{if } x > t.
         \\end{cases}
```
"""
struct SemHinge{B} <: AbstractLoss{ExactHessian}
    threshold::Float64
    param_inds::Vector{Int}   # indices of parameters to regularize
end

"""
    SemHinge(spec::SemSpecification;
             bound = 'l', threshold = 0.0, params)

# Arguments
- `spec`: SEM model specification
- `threshold`: hyperparameter for penalty term
- `params::AbstractVector`: Vector of parameter IDs (Symbols)
  or indices that indicate which parameters should be regularized.

# Examples
```julia
my_hinge = SemHinge(spec; bound = 'u', params = [:λ₁, :λ₂, :ω₂₃])
```
"""
function SemHinge(
    spec::SemSpecification,
    params::AbstractVector;
    bound::Char = 'l',
    threshold::Number = 0.0,
)
    bound ∈ ('l', 'u') || throw(ArgumentError("bound must be either 'l' or 'u', $bound given"))

    param_inds = eltype(params) <: Symbol ? param_indices(spec, params) : params
    return SemHinge{bound}(threshold, param_inds)
end

(hinge::SemHinge{'l'})(val::Number) = max(val - hinge.threshold, 0.0)
(hinge::SemHinge{'u'})(val::Number) = max(hinge.threshold - val, 0.0)

function evaluate!(
    objective, gradient, hessian,
    hinge::SemHinge{B},
    params
) where B
    obj = NaN
    if !isnothing(objective)
        @inbounds obj = sum(i -> hinge(params[i]), hinge.param_inds)
    end
    if !isnothing(gradient)
        fill!(gradient, 0)
        @inbounds for i in hinge.param_inds
            par = params[i]
            if B == 'l' && par > hinge.threshold
                gradient[i] = one(eltype(gradient))
            elseif B == 'u' && par < hinge.threshold
                gradient[i] = -one(eltype(gradient))
            end
        end
    end
    if !isnothing(hessian)
        fill!(hessian, 0)
    end
    return obj
end
