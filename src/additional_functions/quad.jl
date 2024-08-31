# calculate Xᵀ⋅A⋅X
# FIXME: use PDMats.jl when its sparse matrix support is refactored
# see https://github.com/JuliaStats/PDMats.jl/pull/188
function Xt_A_X!(res::AbstractMatrix, A::AbstractMatrix, X::AbstractMatrix,
                 alpha::Real = 1, beta::Real = 0;
                 A_X_buf::Union{AbstractMatrix, Nothing} = nothing)
    A_X = !isnothing(A_X_buf) ? mul!(A_X_buf, A, X) : A * X
    return mul!(res, X', A_X, alpha, beta)
end

function Xt_A_X!(res::AbstractMatrix, A::AbstractMatrix, X::SparseArrays.AbstractSparseMatrixCSC,
                 alpha::Real = 1, beta::Real = 0;
                 Xt_A_buf::Union{AbstractMatrix, Nothing} = nothing)
    Xt_A = !isnothing(Xt_A_buf) ? mul!(Xt_A_buf, X', A) : X'A
    return mul!(res', X', Xt_A', alpha, beta)
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
    return mul!(res, X_A, X', alpha, beta)
end

function X_A_Xt!(res::AbstractMatrix, A::AbstractMatrix, X::SparseArrays.AbstractSparseMatrixCSC,
                 alpha::Real = 1, beta::Real = 0;
                 X_A_buf::Union{AbstractMatrix, Nothing} = nothing)
    X_A = !isnothing(X_A_buf) ? mul!(X_A_buf, X, A) : X * A
    return mul!(res', X, X_A', alpha, beta)
end

X_A_Xt(A::AbstractMatrix, X::AbstractMatrix,
       alpha::Real = 1, beta::Real = 0;
       A_X_buf::Union{AbstractMatrix, Nothing} = nothing) =
    X_A_Xt!(Matrix{promote_type(eltype(A), eltype(X))}(undef, size(X, 1), size(X, 1)),
            A, X, alpha, beta; A_X_buf)
