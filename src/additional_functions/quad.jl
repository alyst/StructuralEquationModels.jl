_unwrap_symmetric(res::AbstractMatrix) = res
_unwrap_symmetric(res::Symmetric) = parent(res)

# internal wrapper for mul!() that can disable some checks for performance reasons
# (i.e. the resulting nz pattern for sparse matrix multiplication -- see SEMMKLSparseExt)
unsafe_mul!(C, A, B, alpha, beta) = mul!(C, A, B, alpha, beta)
unsafe_mul!(C, A, B) = mul!(C, A, B)

# calculate Xᵀ⋅X
Xt_X!(res::AbstractMatrix, X::AbstractMatrix,
      alpha::Real = 1, beta::Real = 0) =
    mul!(_unwrap_symmetric(res), X', X, alpha, beta)

X_Xt!(res::AbstractMatrix, X::Union{AbstractMatrix, AbstractVector},
      alpha::Real = 1, beta::Real = 0) =
    mul!(_unwrap_symmetric(res), X, X', alpha, beta)

Xt_X(X::AbstractMatrix) =
    Xt_X!(Matrix{eltype(X)}(undef, size(X, 2), size(X, 2)), X)

X_Xt(X::Union{AbstractMatrix, AbstractVector}) =
  Xt_X!(Matrix{eltype(X)}(undef, size(X, 1), size(X, 1)), X)

# calculate Xᵀ⋅A⋅X
# FIXME: use PDMats.jl when its sparse matrix support is refactored
# see https://github.com/JuliaStats/PDMats.jl/pull/188
function Xt_A_X!(res::AbstractMatrix, A::AbstractMatrix, X::AbstractMatrix,
                 alpha::Real = 1, beta::Real = 0;
                 Xt_A_buf::Union{AbstractMatrix, Nothing} = nothing)
    Xt_A = !isnothing(Xt_A_buf) ? mul!(Xt_A_buf, X', A) : X'A
    return mul!(_unwrap_symmetric(res), Xt_A, X, alpha, beta)
end

# special handling of symmetric to make sure it is the first argument in *
function Xt_A_X!(res::AbstractMatrix, A::Symmetric{<:Any, M}, X::AbstractMatrix,
                 alpha::Real = 1, beta::Real = 0;
                 Xt_A_buf::Union{AbstractMatrix, Nothing} = nothing) where {M <: StridedMatrix}
    A_X = !isnothing(Xt_A_buf) ? mul!(reshape(Xt_A_buf, size(X)), A, X) : A*X
    return mul!(_unwrap_symmetric(res), X', A_X, alpha, beta)
end

Xt_A_X(A::AbstractMatrix, X::AbstractMatrix,
       alpha::Real = 1, beta::Real = 0;
       A_X_buf::Union{AbstractMatrix, Nothing} = nothing) =
    Xt_A_X!(Matrix{promote_type(eltype(A), eltype(X))}(undef, size(X, 2), size(X, 2)),
            A, X, alpha, beta; A_X_buf)

function X_A_Xt!(res::AbstractMatrix, A::AbstractMatrix, X::AbstractMatrix,
                 alpha::Real = 1, beta::Real = 0;
                 X_A_buf::Union{AbstractMatrix, Nothing} = nothing)
    X_A = !isnothing(X_A_buf) ? mul!(X_A_buf, X, A) : X * A
    return mul!(_unwrap_symmetric(res), X_A, X', alpha, beta)
end

# special handling of symmetric to make sure it is the first argument in *
function X_A_Xt!(res::AbstractMatrix, A::Symmetric{<:Any, M}, X::AbstractMatrix,
                 alpha::Real = 1, beta::Real = 0;
                 X_A_buf::Union{AbstractMatrix, Nothing} = nothing) where {M <: StridedMatrix}
    A_Xt = !isnothing(X_A_buf) ? mul!(reshape(X_A_buf, size(X, 2), size(X, 1)), A, X') : A * X'
    return mul!(_unwrap_symmetric(res), X, A_Xt, alpha, beta)
end

X_A_Xt(A::AbstractMatrix, X::AbstractMatrix,
       alpha::Real = 1, beta::Real = 0;
       X_A_buf::Union{AbstractMatrix, Nothing} = nothing) =
    X_A_Xt!(Matrix{promote_type(eltype(A), eltype(X))}(undef, size(X, 1), size(X, 1)),
            A, X, alpha, beta; X_A_buf)
