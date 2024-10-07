module SEMMKLSparseExt

import StructuralEquationModels: Xt_X!, X_Xt!, Xt_A_X!, X_A_Xt!
using StructuralEquationModels: _unwrap_symmetric, blascopytri!, fastcopytri!
using SparseArrays, LinearAlgebra
using MKLSparse: syrkd!, syprd!, SparseMatrixCSR

# overload unsafe_mul!(sparse, sparse, sparse) with nzpattern checks disabled
# (since within the fixed SEM model the nzpattern does not change)
unsafe_mul!(C::AbstractSparseMatrix, A, B, alpha, beta) =
    MKLSparse.unsafe_mul!(C, A, B, alpha, beta; check_nzpattern = false)
unsafe_mul!(C::AbstractSparseMatrix, A, B) =
    MKLSparse.unsafe_mul!(C, A, B; check_nzpattern = false)

# calculate Xᵀ⋅X
function Xt_X!(res::AbstractMatrix{T}, X::AbstractSparseMatrix{T},
               alpha::Real = 1, beta::Real = 0;
               check::Bool = true
) where T
    if iszero(beta) || (!check || issymmetric(res))
        syrkd!('N', T(alpha), convert(SparseMatrixCSR, transpose(X)),
            T(beta), _unwrap_symmetric(res))
        fastcopytri!(res, 'U')
    else # generic sparse*sparse mul
        mul!(res, transpose(X), X, alpha, beta)
        #@assert issymmetric(res)
    end
    return res
end

function X_Xt!(res::AbstractMatrix{T}, X::AbstractSparseMatrix{T},
               alpha::Real = 1, beta::Real = 0;
               check::Bool = true
) where T
    if iszero(beta) || (!check || issymmetric(res))
        syrkd!('T', T(alpha), convert(SparseMatrixCSR, transpose(X)),
            T(beta), _unwrap_symmetric(res))
        fastcopytri!(res, 'U')
    else # generic sparse*sparse mul
        mul!(res, X, transpose(X), alpha, beta)
        #@assert issymmetric(res)
    end
end

# calculate Xᵀ⋅A⋅X
function Xt_A_X!(res::AbstractMatrix{T}, A::StridedMatrix{T},
                 X::AbstractSparseMatrix{T},
                 alpha::Real = 1, beta::Real = 0;
                 Xt_A_buf::Union{AbstractMatrix, Nothing} = nothing
) where T
    syprd!('N', T(alpha), convert(SparseMatrixCSR, transpose(X)),
           A, T(beta), _unwrap_symmetric(res))
    fastcopytri!(res, 'U')
end

Xt_A_X!(res::AbstractMatrix{T}, A::Symmetric{T, <:StridedMatrix{T}},
        X::AbstractSparseMatrix{T},
        alpha::Real = 1, beta::Real = 0;
        Xt_A_buf::Union{AbstractMatrix, Nothing} = nothing
) where T =
    Xt_A_X!(res, _unwrap_symmetric(A), X, alpha, beta; Xt_A_buf)

function X_A_Xt!(res::AbstractMatrix{T}, A::StridedMatrix{T},
                 X::AbstractSparseMatrix{T},
                 alpha::Real = 1, beta::Real = 0;
                 X_A_buf::Union{AbstractMatrix, Nothing} = nothing
) where T
    syprd!('T', T(alpha), convert(SparseMatrixCSR, transpose(X)),
           _unwrap_symmetric(A), T(beta), _unwrap_symmetric(res))
    fastcopytri!(res, 'U')
end

X_A_Xt!(res::AbstractMatrix{T}, A::Symmetric{T, <:StridedMatrix{T}},
        X::AbstractSparseMatrix{T},
        alpha::Real = 1, beta::Real = 0;
        X_A_buf::Union{AbstractMatrix, Nothing} = nothing
) where T =
    X_A_Xt!(res, _unwrap_symmetric(A), X, alpha, beta; X_A_buf)

end