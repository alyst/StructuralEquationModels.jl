# Neumann seriess representation of (I - mat)⁻¹
function neumann_series(mat::SparseMatrixCSC;
                        maxn::Integer = size(mat, 1),
                        Aⁿ_rewriter = nothing)
    inverse = I
    next_term = mat

    n = 0
    converged = false
    while n < maxn
        converged = nnz(next_term) == 0
        converged && break
        n += 1
        inverse += next_term
        next_term *= mat
        if !isnothing(Aⁿ_rewriter)
            for i in eachindex(next_term.nzval)
                next_term.nzval[i] = Aⁿ_rewriter(next_term.nzval[i])
            end
            dropzeros!(next_term)
        end
    end
    if !converged
        @warn("Neumann series did not converge in $maxn steps")
    end

    return inverse
end

#=
function make_onelement_array(A)
    isa(A, Array) ? nothing : (A = [A])
    return A
end
 =#

function semvec(observed, imply, loss, optimizer)

    observed = make_onelement_array(observed)
    imply = make_onelement_array(imply)
    loss = make_onelement_array(loss)
    optimizer = make_onelement_array(optimizer)

    #sem_vec = Array{AbstractSem}(undef, maximum(length.([observed, imply, loss, optimizer])))
    sem_vec = Sem.(observed, imply, loss, optimizer)

    return sem_vec
end

# construct a vector of SemObserved objects
# for each specified data row
function observed(::Type{T}, data, rowinds;
            args = (),
            kwargs = NamedTuple()) where T <: SemObserved
    return T[
        T(args...;
          data = Matrix(view(data, row, :)),
          kwargs...)
        for row in rowinds]
end

function skipmissing_mean(mat::AbstractMatrix)
    means = [mean(skipmissing(coldata))
             for coldata in eachcol(mat)]
    return means
end

function F_one_person(imp_mean, meandiff, inverse, data, logdet)
    F = logdet
    @. meandiff = data - imp_mean
    F += dot(meandiff, inverse, meandiff)
    return F
end

function remove_all_missing(data::AbstractMatrix)
    keep = Vector{Int64}()
    for (i, coldata) in zip(axes(data, 1), eachrow(data))
        if any(!ismissing, coldata)
            push!(keep, i)
        end
    end
    return data[keep, :], keep
end

#=
function batch_sym_inv_update!(fun::Union{LossFunction, DiffFunction}, model)
    M_inv = inv(fun.choleskys[1])
    for i = 1:size(fun.inverses, 1)
        if size(model.observed.patterns_not[i]) == 0
            fun.inverses[i] .= M_inv
        else
            ind_not = model.observed.patterns_not[i]
            ind = model.observed.patterns[i]

            A = M_inv[ind_not, ind]
            H = cholesky(M_inv[ind_not, ind_not])
            D = H \ A
            out = M_inv[ind, ind] - LinearAlgebra.BLAS.gemm('T', 'N', 1.0, A, D)
            fun.inverses[i] .= out
        end
    end
end =#

function sparse_outer_mul!(C, A, B, ind) #computes A*S*B -> C, where ind gives the entries of S that are 1
    fill!(C, 0.0)
    for i in 1:length(ind)
        BLAS.ger!(1.0, A[:, ind[i][1]], B[ind[i][2], :], C)
    end
end

function sparse_outer_mul!(C, A, ind) #computes A*∇m, where ∇m ind gives the entries of ∇m that are 1
    fill!(C, 0.0)
    @views C .= sum(A[:, ind], dims = 2)
    return C
end

function sparse_outer_mul!(C, A, B::Vector, ind) #computes A*S*B -> C, where ind gives the entries of S that are 1
    fill!(C, 0.0)
    @views @inbounds for i in 1:length(ind)
        C .+= B[ind[i][2]].*A[:, ind[i][1]]
    end
end

# n²×(n(n+1)/2) matrix to transform a vector of lower
# triangular entries into a vectorized form of a n×n symmetric matrix,
# opposite of elimination_matrix()
function duplication_matrix(n::Integer)
    ntri = div(n*(n+1), 2)
    D = zeros(n^2, ntri)
    for j in 1:n
        for i in j:n
            tri_ix = (j-1)*n + i - div(j*(j-1), 2)
            D[j + n*(i-1), tri_ix] = 1
            D[i + n*(j-1), tri_ix] = 1
        end
    end
    return D
end

# (n(n+1)/2)×n² matrix to transform a
# vectorized form of a n×n symmetric matrix
# into vector of its lower triangular entries,
# opposite of duplication_matrix()
function elimination_matrix(n::Integer)
    ntri = div(n*(n+1), 2)
    L = zeros(ntri, n^2)
    for j in 1:n
        for i in j:n
            tri_ix = (j-1)*n + i - div(j*(j-1), 2)
            L[tri_ix, i + n*(j-1)] = 1
        end
    end
    return L
end

# vector of lower-triangular values of a square matrix
function vech(A::AbstractMatrix{T}) where T
    size(A, 1) == size(A, 2) ||
        throw(ArgumentError("Matrix must be square, $(size(A)) given"))
    n = size(A, 1)
    v = Vector{T}(undef, (n * (n + 1)) >> 1)
    k = 0
    for (j, Aj) in enumerate(eachcol(A)), i in j:n
        @inbounds v[k += 1] = Aj[i]
    end
    @assert k == length(v)
    return v
end

# vector of lower-triangular linear indices of a nXn square matrix
function vechinds(n::Integer)
    A_lininds = LinearIndices((n, n))
    v = Vector{Int}(undef, (n * (n + 1)) >> 1)
    k = 0
    for j in 1:n, i in j:n
        @inbounds v[k += 1] = A_lininds[i, j]
    end
    @assert k == length(v)
    return v
end

# vector of lower-triangular linear indices of a square matrix
function vechinds(A::AbstractMatrix)
    size(A, 1) == size(A, 2) ||
        throw(ArgumentError("Matrix must be square, $(size(A)) given"))
    return vechinds(size(A, 1))
end

# returns the vector of non-unique values in the order of appearance
# each non-unique values is reported once
function nonunique(values::AbstractVector)
    value_counts = Dict{eltype(values), Int}()
    res = similar(values, 0)
    for v in values
        n = get!(value_counts, v, 0)
        if n == 1 # second encounter
            push!(res, v)
        end
        value_counts[v] = n + 1
    end
    return res
end

# truncate eigenvalues of a symmetric matrix and return the result
function trunc_eigvals(mtx::AbstractMatrix{T}, min_eigval::Number;
                       mtx_label::AbstractString = "matrix",
                       verbose::Bool = false
) where T
    # eigen decomposition of the mtx
    mtx_eig = eigen(convert(Matrix{T}, mtx))
    verbose && @info "min(eigvals($mtx_label))=$(Base.minimum(mtx_eig.values)), N(eigvals < $min_eigval) = $(sum(<(min_eigval), mtx_eig.values))"

    # substitute small eigvals with min_eigval
    newmtx = X_A_Xt(Diagonal(max.(mtx_eig.values, min_eigval)), mtx_eig.vectors)
    StatsBase._symmetrize!(newmtx)
    if verbose
        Δmtx = newmtx .- mtx
        @info "Δ($mtx_label, posdef)=$(norm(Δmtx, 2)), min,max(Δᵢ)=$(extrema(Δmtx))"
    end

    return newmtx
end

function index_in_sorted(v::Union{AbstractVector, Colon}, x)
    (v isa Colon) && return x
    i = searchsortedfirst(v, x)
    return i <= lastindex(v) && v[i] == x ? i : nothing
end

function check_subaxis(arr::AbstractArray, subarr::AbstractArray, axis::Integer, subinds::Union{AbstractVector, Colon})
    if isa(subinds, Colon)
        size(arr, axis) == size(subarr, axis) ||
            throw(DimensionMismatch("submtx $(axis == 1 ? "rows" : "columns") ($(size(subarr, axis))) does not match the matrix $(axis == 1 ? "rows" : "columns") ($(size(arr, axis)))"))
    else
        length(subinds) == size(subarr, axis) ||
            throw(DimensionMismatch("submtx $(axis == 1 ? "rows" : "columns") ($(size(subarr, axis))) does not match the subindices length ($(length(subinds)))"))
        issorted(subinds) || throw(ArgumentError("subindices must be sorted"))
    end
end

# create the mapping between the indices of the nonzero values of the sparse submatrix
# and the indices of the nonzero values of the full matrix
# return the vector of nzval indices of mtx that correspond to the submtx.nzval
function nzsubmatrix_to_nzmatrix(mtx::SparseMatrixCSC, submtx::SparseMatrixCSC,
                                 rowinds::Union{AbstractVector, Colon},
                                 colinds::Union{AbstractVector, Colon}
)
    check_subaxis(mtx, submtx, 1, rowinds)
    check_subaxis(mtx, submtx, 2, colinds)

    inds = Vector{Int}()
    nzrows, nzcols, _ = findnz(mtx)
    for (nzind, (i, j)) in enumerate(zip(nzrows, nzcols))
        if !isnothing(index_in_sorted(rowinds, i)) &&
           !isnothing(index_in_sorted(colinds, j))
            push!(inds, nzind)
        end
    end
    length(inds) != nnz(submtx) &&
        throw(ArgumentError("Non-zeros count in submatrix ($(nnz(submtx))) does not match the nonzeros in the full matrix subset ($(length(inds)))"))
    return inds
end

# return the tuple of linear indices of the dense submtx
# and the correspondingg indices of mtx.nzval entries
function nzsubmatrix_to_nzmatrix(mtx::SparseMatrixCSC, submtx::StridedMatrix,
                                 rowinds::Union{AbstractVector, Colon},
                                 colinds::Union{AbstractVector, Colon}
)
    check_subaxis(mtx, submtx, 1, rowinds)
    check_subaxis(mtx, submtx, 2, colinds)

    sub_lininds = LinearIndices(submtx)
    nzrows, nzcols, _ = findnz(mtx)
    srcinds = Vector{Int}()
    destinds = Vector{Int}()

    for (nzind, (i, j)) in enumerate(zip(nzrows, nzcols))
        subi = index_in_sorted(rowinds, i)
        isnothing(subi) && continue
        subj = index_in_sorted(colinds, j)
        isnothing(subj) && continue
        push!(srcinds, sub_lininds[subi, subj])
        push!(destinds, nzind)
    end
    return srcinds, destinds
end
