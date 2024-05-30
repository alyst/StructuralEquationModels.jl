"""
    SemHinge{B} <: SemLossFunction{ExactHessian}

Hinge regularization.

Implements *squared hinge* a.k.a *squared rectified linear unit* (*SReLU*) loss function:
```math
f_{t}(x) = \\begin{cases} 0 & \\text{if}\\ x \\leq t \\\\
                        (x - t)^2 & \\text{if } x > t.
         \\end{cases}
```
"""
struct SemSquaredHinge{B} <: AbstractLoss{ExactHessian}
    threshold::Float64
    param_inds::Vector{Int}   # indices of parameters to regularize
    H_diag_inds::Vector{Int} # indices of Hessian diagonal elements to regularize
end

"""
    SemSquaredHinge(spec::SemSpecification;
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
function SemSquaredHinge(
    spec::SemSpecification,
    params::AbstractVector;
    bound::Char = 'l',
    threshold::Number = 0.0
)
    bound ∈ ('l', 'u') || throw(ArgumentError("bound must be either 'l' or 'u', $bound given"))

    param_inds = eltype(params) <: Symbol ? param_indices(spec, params) : params
    H_linind = LinearIndices((nparams(spec), nparams(spec)))
    return SemSquaredHinge{bound}(threshold, param_inds,
                                  [H_linind[i, i] for i in param_inds])
end

(sqhinge::SemSquaredHinge{'l'})(val::Number) = abs2(max(val - sqhinge.threshold, 0.0))
(sqhinge::SemSquaredHinge{'u'})(val::Number) = abs2(max(sqhinge.threshold - val, 0.0))

function evaluate!(
    objective, gradient, hessian,
    sqhinge::SemSquaredHinge{B},
    params
) where B
    obj = NaN
    if !isnothing(objective)
        @inbounds obj = sum(i -> sqhinge(params[i]), sqhinge.param_inds)
    end
    if !isnothing(gradient)
        fill!(gradient, 0)
        @inbounds for i in sqhinge.param_inds
            par = params[i]
            if (B == 'l' && par > sqhinge.threshold) ||
               (B == 'u' && par < sqhinge.threshold)
                gradient[i] = 2 * (par - sqhinge.threshold)
            end
        end
    end
    if !isnothing(hessian)
        fill!(hessian, 0)
        @inbounds for (par_i, H_i) in zip(sqhinge.param_inds, sqhinge.H_diag_inds)
            par = params[par_i]
            if (B == 'l' && par > sqhinge.threshold) ||
               (B == 'u' && par < sqhinge.threshold)
                hessian[H_i] = 2
            end
        end
    end
    return obj
end
