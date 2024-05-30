############################################################################################
# Define the basic type system
############################################################################################

"Meanstructure trait for `SemImply` subtypes"
abstract type MeanStructure end
"Indicates that `SemImply` subtype supports meanstructure"
struct HasMeanStructure <: MeanStructure end
"Indicates that `SemImply` subtype does not support meanstructure"
struct NoMeanStructure <: MeanStructure end

# fallback implementation
MeanStructure(::Type{T}) where T = error("Objects of type $T do not support MeanStructure trait")
MeanStructure(semobj) = MeanStructure(typeof(semobj))

"Hessian Evaluation trait for `SemImply` and `SemLossFunction` subtypes"
abstract type HessianEvaluation end
struct ApproximateHessian <: HessianEvaluation end
struct ExactHessian <: HessianEvaluation end

# fallback implementation
HessianEvaluation(::Type{T}) where T = error("Objects of type $T do not support HessianEvaluation trait")
HessianEvaluation(semobj) = HessianEvaluation(typeof(semobj))

"Supertype for all loss functions of SEMs. If you want to implement a custom loss function, it should be a subtype of `AbstractLoss`."
abstract type AbstractLoss{HE <: HessianEvaluation} end

HessianEvaluation(::Type{<:AbstractLoss{HE}}) where HE <: HessianEvaluation = HE

"""
Supertype of all objects that can serve as the `optimizer` field of a SEM.
Connects the SEM to its optimization backend and controls options like the optimization algorithm.
If you want to connect the SEM package to a new optimization backend, you should implement a subtype of SemOptimizer.
"""
abstract type SemOptimizer{E} end

engine(::Type{SemOptimizer{E}}) where E = E
engine(optimizer::SemOptimizer) = engine(typeof(optimizer))

SemOptimizer(args...; engine::Symbol = :Optim, kwargs...) =
    SemOptimizer{engine}(args...; kwargs...)

# fallback optimizer constructor
function SemOptimizer{E}(args...; kwargs...) where E
    throw(ErrorException("$E optimizer is not supported."))
end

"""
Supertype of all objects that can serve as the observed field of a SEM.
Pre-processes data and computes sufficient statistics for example.
If you have a special kind of data, e.g. ordinal data, you should implement a subtype of SemObserved.
"""
abstract type SemObserved end

get_data(observed::SemObserved) = observed.data
observed_vars(observed::SemObserved) = observed.observed_vars

"""
Supertype of all objects that can serve as the imply field of a SEM.
Computed model-implied values that should be compared with the observed data to find parameter estimates,
e. g. the model implied covariance or mean.
If you would like to implement a different notation, e.g. LISREL, you should implement a subtype of SemImply.
"""
abstract type SemImply{MS <: MeanStructure, HE <: HessianEvaluation} end

MeanStructure(::Type{<:SemImply{MS}}) where MS <: MeanStructure = MS
HessianEvaluation(::Type{<:SemImply{MS,HE}}) where {MS, HE <: MeanStructure} = HE

"Subtype of SemImply for all objects that can serve as the imply field of a SEM and use some form of symbolic precomputation."
abstract type SemImplySymbolic{MS,HE} <: SemImply{MS,HE} end

"""
State of `SemImply` that corresponds to the specific SEM parameter values.

Contains the necessary vectors and matrices for calculating the SEM
objective, gradient and hessian (whichever is requested).
"""
abstract type SemImplyState end

imply(state::SemImplyState) = state.imply
MeanStructure(state::SemImplyState) = MeanStructure(imply(state))
ApproximateHessian(state::SemImplyState) = ApproximateHessian(imply(state))

"""
    abstract type SemLoss{O <: SemObserved, I <: SemImply, HE <: HessianEvaluation} <: AbstractLoss{HE} end

The base type for calculating the loss of the implied SEM model when explaining the observed data.

All subtypes of `SemLoss` should have the following fields:
- `observed::O`: object of subtype [`SemObserved`](@ref).
- `imply::I`: object of subtype [`SemImply`](@ref).
"""
abstract type SemLoss{O <: SemObserved, I <: SemImply, HE <: HessianEvaluation} <: AbstractLoss{HE} end

"Most abstract supertype for all SEMs"
abstract type AbstractSem end

"[`AbstractLoss`](@ref) as a weighted term in a [`Sem`](@ref) model"
struct LossTerm{L <: AbstractLoss, I <: Union{Symbol, Nothing}, W <: Union{Number, Nothing}}
    loss::L
    id::I
    weight::W
end

"""
    Sem(loss_terms...; [params], kwargs...)

SEM model (including model ensembles) that combines all the data, implied SEM structure
and regularization terms and implements the calculation of their weighted sum, as well as its
gradient and (optionally) Hessian.

# Arguments
- `loss_terms...`: [`AbstractLoss`](@ref) objects, including SEM losses ([`SemLoss`](@ref)),
  optionally can be a pair of a loss object and its numeric weight

# Fields
- `loss_terms::Tuple`: a tuple of all loss functions and their weights
- `params::Vector{Symbol}`: the vector of parameter ids shared by all loss functions.
"""
struct Sem{L <: Tuple} <: AbstractSem
    loss_terms::L
    params::Vector{Symbol}
end

############################################################################################
# automatic differentiation
############################################################################################

"""
    SemFiniteDiff(;observed = SemObservedData, imply = RAM, loss = SemML, kwargs...)

A wrapper around [`Sem`](@ref) that substitutes dedicated evaluation of gradient and hessian with
finite difference approximation.

# Arguments
- `model::Sem`: the SEM model to wrap
"""
struct SemFiniteDiff{S <: AbstractSem} <: AbstractSem
    model::S
end

_unwrap(wrapper::SemFiniteDiff) = wrapper.model
params(wrapper::SemFiniteDiff) = params(wrapper.model)
loss_terms(wrapper::SemFiniteDiff) = loss_terms(wrapper.model)

struct LossFiniteDiff{L <: AbstractLoss} <: AbstractLoss{ApproximateHessian}
    loss::L
end

struct SemLossFiniteDiff{O, I, L <: SemLoss{O, I}} <: SemLoss{O, I, ApproximateHessian}
    loss::L
end

FiniteDiffLossWrappers = Union{LossFiniteDiff, SemLossFiniteDiff}

_unwrap(term::AbstractLoss) = term
_unwrap(wrapper::FiniteDiffLossWrappers) = wrapper.loss
imply(wrapper::FiniteDiffLossWrappers) = imply(_unwrap(wrapper))
observed(wrapper::FiniteDiffLossWrappers) = observed(_unwrap(wrapper))

FiniteDiffWrapper(model::AbstractSem) = SemFiniteDiff(model)
FiniteDiffWrapper(loss::AbstractLoss) = LossFiniteDiff(loss)
FiniteDiffWrapper(loss::SemLoss) = SemLossFiniteDiff(loss)

abstract type SemSpecification end

abstract type AbstractParameterTable <: SemSpecification end
