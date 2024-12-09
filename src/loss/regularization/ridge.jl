# (Ridge) regularization

############################################################################################
### Types
############################################################################################
"""
Ridge regularization.

    struct SemRidge{T, TB} <: AbstractLoss{ExactHessian}

Ridge regularization is a type of *L²* regularization that penalizes the sum of squared parameter values.
`SemRidge` also allows specifying an optional affine transform (*A × p + b*)
to apply to the parameters before the regularization.

# Constructors
    SemRidge(A::SparseMatrixCSC, b::Union{AbstractVector, Nothing} = nothing)
    SemRidge(spec::SemSpecification, params::AbstractVector,
             [A::AbstractMatrix = nothing],
             [b::AbstractVector = nothing])

# Arguments
- `spec`: SEM model specification.
- `params::Vector`: optional IDs (Symbols) or indices of parameters to regularize.
- `A`: optional transformation matrix that defines how to transform the vector of parameter values
       before the regularization. If `params` is not specified, the transformation is applied
       to the entire parameters vector.
- `b`: optional vector of intercepts to add to the transformed parameters.

# Examples
```julia
my_ridge = SemRidge(spec, [:λ₁, :λ₂, :ω₂₃])
my_trans_ridge = SemRidge(spec, [:λ₁, :λ₂, :ω₂₃], [1.0 1.0 0.0; 0.0 0.0 1.0], [-2.0, 0.0])
```
"""
struct SemRidge{T, TB} <: AbstractLoss{ExactHessian}
    A::SparseMatrixCSC{T, Int}      # transformation/subsetting of the parameters
    Atx2::SparseMatrixCSC{T, Int}   # 2Aᵀ
    b::TB                           # optional transformed parameter intercepts
    H_inds::Vector{Int}             # non-zero linear indices of Hessian
    H_vals::Vector{T}               # non-zero values of Hessian
end

############################################################################
### Constructors
############################################################################

function SemRidge(
    A::SparseMatrixCSC,
    b::Union{AbstractVector, Nothing} = nothing
)
    isnothing(b) || length(b) == size(A, 1) ||
        throw(DimensionMismatch("The intercept length ($(length(b))) should match the rows of " *
                                "the transformation matrix ($(size(A, 1)))"))
    Atx2 = convert(typeof(A), A')
    Atx2 .*= 2
    H = Atx2 * A
    H_inds = LinearIndices(size(H))
    H_i, H_j, H_v = findnz(H)
    return SemRidge{eltype(A), typeof(b)}(
                A, Atx2, b,
                [H_inds[i, j] for (i, j) in zip(H_i, H_j)], copy(H_v))
end

function SemRidge(
    spec::SemSpecification,
    params::AbstractVector,
    A::Union{AbstractMatrix, Nothing} = nothing,
    b::Union{AbstractVector, Nothing} = nothing
)
    param_inds = eltype(params) <: Symbol ? param_indices(spec, params) : params

    isnothing(A) || size(A, 2) == length(param_inds) ||
        throw(DimensionMismatch("The transformation matrix columns ($(size(A, 2))) should match " *
                                "the parameters to regularize ($(length(param_inds)))"))

    sel_params_mtx = eachrow_to_col(Float64, param_inds, nparams(spec))
    spA = !isnothing(A) ? convert(SparseMatrixCSC, A * sel_params_mtx) : sel_params_mtx
    return SemRidge(spA, b)
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
    trf_params = ridge.A * params
    if !isnothing(ridge.b)
        trf_params .+= ridge.b
    end
    if !isnothing(objective)
        obj = sum(abs2, trf_params)
    end
    if !isnothing(gradient)
        mul!(gradient, ridge.Atx2, trf_params)
    end
    if !isnothing(hessian)
        fill!(hessian, 0)
        @inbounds hessian[ridge.H_inds] .= ridge.H_vals
    end
    return obj
end

############################################################################################
### Recommended methods
############################################################################################

update_observed(loss::SemRidge, observed::SemObserved; kwargs...) = loss