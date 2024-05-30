# constant loss function for comparability with other packages

############################################################################################
### Types
############################################################################################
"""
    SemConstant{C <: Number} <: AbstractLoss{ExactHessian}

Constant loss term. Can be used for comparability to other packages.

# Constructor

    SemConstant(;constant_loss, kwargs...)

# Arguments
- `constant_loss::Number`: constant to add to the objective

# Examples
```julia
    my_constant = SemConstant(42.0)
```
"""
struct SemConstant{C <: Number} <: AbstractLoss{ExactHessian}
    c::C
end

objective(constant::SemConstant, par) = constant.c
gradient(constant::SemConstant, par) = zero(par)
hessian(constant::SemConstant, par) = zeros(eltype(par), length(par), length(par))

############################################################################################
### Recommended methods
############################################################################################

update_observed(loss_function::SemConstant, observed::SemObserved; kwargs...) = loss_function