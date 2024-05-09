
############################################################################################
### Type
############################################################################################

struct RAMMatrices <: SemSpecification
    A::ParamsMatrix{Float64}
    S::ParamsMatrix{Float64}
    F::SparseMatrixCSC{Float64}
    M::Union{ParamsVector{Float64}, Nothing}
    params::Vector{Symbol}
    colnames::Union{Vector{Symbol}, Nothing}    # better call it "variables": it's a mixture of observed and latent (and it gets confusing with get_colnames())
end

nparams(ram::RAMMatrices) = nparams(ram.A)
nvars(ram::RAMMatrices) = size(ram.F, 2)
nobserved_vars(ram::RAMMatrices) = size(ram.F, 1)
nlatent_vars(ram::RAMMatrices) = nvars(ram) - nobserved_vars(ram)

vars(ram::RAMMatrices) = ram.colnames

isobserved_var(ram::RAMMatrices, i::Integer) =
    ram.F.colptr[i+1] > ram.F.colptr[i]
islatent_var(ram::RAMMatrices, i::Integer) =
    ram.F.colptr[i+1] == ram.F.colptr[i]

observed_var_indices(ram::RAMMatrices) =
    [i for i in axes(ram.F, 2) if isobserved_var(ram, i)]
latent_var_indices(ram::RAMMatrices) =
    [i for i in axes(ram.F, 2) if islatent_var(ram, i)]

# observed variables in the order as they appear in ram.F rows
function observed_vars(ram::RAMMatrices)
    if isnothing(ram.colnames)
        @warn "Your RAMMatrices do not contain column names. Please make sure the order of variables in your data is correct!"
        return nothing
    else
        return ram.colnames[ram.F.rowval]
    end
end

function latent_vars(ram::RAMMatrices)
    if isnothing(ram.colnames)
        @warn "Your RAMMatrices do not contain column names. Please make sure the order of variables in your data is correct!"
        return nothing
    else
        return [col for (i, col) in enumerate(ram.colnames)
                if islatent_var(ram, i)]
    end
end

function variance_params(ram::RAMMatrices)
    S_diaginds = Set(diagind(ram.S))
    varparams = Vector{Symbol}()
    for (i, param) in enumerate(ram.params)
        if any(∈(S_diaginds), param_occurences(ram.S, i))
            push!(varparams, param)
        end
    end
    return unique!(varparams)
end

############################################################################################
### Constructor
############################################################################################

function RAMMatrices(; A::AbstractMatrix, S::AbstractMatrix,
                       M::Union{AbstractVector, Nothing} = nothing,
                       F::Union{AbstractMatrix, Nothing} = nothing,
                     observed_vars::Union{AbstractVector{Symbol}, Nothing} = nothing,
                     params::AbstractVector{Symbol},
                     colnames::Union{AbstractVector{Symbol}, Nothing} = nothing)
    ncols = size(A, 2)
    if !isnothing(colnames)
        length(colnames) == ncols || throw(DimensionMismatch("colnames length ($(length(colnames))) does not match the number of columns in A ($ncols)"))
        dup_cols = nonunique(colnames)
        isempty(dup_cols) || throw(ArgumentError("Duplicate variables detected: $(join(dup_cols, ", "))"))
    end
    if !isnothing(observed_vars)
        dup_rows = nonunique(observed_vars)
        isempty(dup_cols) || throw(ArgumentError("Duplicate observed variables detected: $(join(dup_rows, ", "))"))
    end
    size(A, 1) == size(A, 2) || throw(DimensionMismatch("A must be a square matrix"))
    size(S, 1) == size(S, 2) || throw(DimensionMismatch("S must be a square matrix"))
    size(A, 2) == ncols || throw(DimensionMismatch("A should have as many rows and columns as colnames length ($ncols), $(size(A)) found"))
    size(S, 2) == ncols || throw(DimensionMismatch("S should have as many rows and columns as colnames length ($ncols), $(size(S)) found"))
    if !isnothing(F)
        if !isnothing(observed_vars)
            size(F, 1) == length(observed_vars) || throw(DimensionMismatch("F should have as many rows as observed variables ($(length(observed_vars))), $(size(F, 1)) found"))
        else # generated observed vars
        end
        size(F, 2) == ncols || throw(DimensionMismatch("F should have as many columns as colnames length ($ncols), $(size(F, 2)) found"))
    else # generate F using colnames and observed_vars
    end
    if !isnothing(M)
        length(M) == ncols || throw(DimensionMismatch("M should have as many elements as colnames length ($ncols), $(length(M)) found"))
    end
    dup_params = nonunique(params)
    isempty(dup_params) || throw(ArgumentError("Duplicate parameters detected: $(join(dup_params, ", "))"))

    A = ParamsMatrix{Float64}(A, params)
    S = ParamsMatrix{Float64}(S, params)
    M = !isnothing(M) ? ParamsVector{Float64}(M, params) : nothing
    spF = sparse(F)
    if any(!isone, spF.nzval)
        throw(ArgumentError("F should contain only 0s and 1s"))
    end
    return RAMMatrices(A, S, F, M, copy(params), colnames)
end

############################################################################################
### get RAMMatrices from parameter table
############################################################################################

function RAMMatrices(partable::ParameterTable;
                     params::Union{AbstractVector{Symbol}, Nothing} = nothing)

    params = copy(isnothing(params) ? SEM.params(partable) : params)
    dup_params = nonunique(params)
    isempty(dup_params) || throw(ArgumentError("Duplicate parameters detected: $(join(dup_params, ", "))"))

    params_index = Dict(param => i for (i, param) in enumerate(params))
    n_observed = length(partable.variables.observed)
    n_latent = length(partable.variables.latent)
    n_vars = n_observed + n_latent

    # colnames (variables)
    # and F indices (map from each observed column to its variable index)
    if length(partable.variables.sorted) != 0
        @assert length(partable.variables.sorted) == nvars(partable)
        colnames = copy(partable.variables.sorted)
        F_inds = findall(∈(Set(partable.variables.observed)),
                         colnames)
    else
        colnames = [partable.variables.observed;
                    partable.variables.latent]
        F_inds = 1:n_observed
    end

    # indices of the colnames
    cols_index = Dict(col => i for (i, col) in enumerate(colnames))

    # fill Matrices
    # known_labels = Dict{Symbol, Int64}()

    T = nonmissingtype(eltype(partable.columns.value_fixed))
    A_inds = [Vector{Int64}() for _ in 1:length(params)]
    A_lin_ixs = LinearIndices((n_vars, n_vars))
    S_inds = [Vector{Int64}() for _ in 1:length(params)]
    S_lin_ixs = LinearIndices((n_vars, n_vars))
    A_consts = Vector{Pair{Int, T}}()
    S_consts = Vector{Pair{Int, T}}()
    # is there a meanstructure?
    M_inds = any(==(Symbol("1")), partable.columns.from) ?
        [Vector{Int64}() for _ in 1:length(params)] : nothing
    M_consts = !isnothing(M_inds) ? Vector{Pair{Int, T}}() : nothing

    for row in partable

        row_ind = cols_index[row.to]
        col_ind = row.from != Symbol("1") ? cols_index[row.from] : nothing

        if !row.free
            if (row.relation == :→) && (row.from == Symbol("1"))
                push!(M_consts, row_ind => row.value_fixed)
            elseif (row.relation == :→)
                push!(A_consts, A_lin_ixs[CartesianIndex(row_ind, col_ind)] => row.value_fixed)
            elseif (row.relation == :↔)
                push!(S_consts, S_lin_ixs[CartesianIndex(row_ind, col_ind)] => row.value_fixed)
                if row_ind != col_ind # symmetric
                    push!(S_consts, S_lin_ixs[CartesianIndex(col_ind, row_ind)] => row.value_fixed)
                end
            else
                error("Unsupported relation: $(row.relation)")
            end
        else
            par_ind = params_index[row.param]
            if (row.relation == :→) && (row.from == Symbol("1"))
                push!(M_inds[par_ind], row_ind)
            elseif row.relation == :→
                push!(A_inds[par_ind], A_lin_ixs[CartesianIndex(row_ind, col_ind)])
            elseif row.relation == :↔
                push!(S_inds[par_ind], S_lin_ixs[CartesianIndex(row_ind, col_ind)])
                if row_ind != col_ind # symmetric
                    push!(S_inds[par_ind], S_lin_ixs[CartesianIndex(col_ind, row_ind)])
                end
            else
                error("Unsupported relation: $(row.relation)")
            end
        end
    end
    # sort linear indices
    for A_ind in A_inds
        sort!(A_ind)
    end
    for S_ind in S_inds
        unique!(sort!(S_ind)) # also symmetric duplicates
    end
    if !isnothing(M_inds)
        for M_ind in M_inds
            sort!(M_ind)
        end
    end
    sort!(A_consts, by=first)
    sort!(S_consts, by=first)
    if !isnothing(M_consts)
        sort!(M_consts, by=first)
    end

    return RAMMatrices(ParamsMatrix{T}(A_inds, A_consts, (n_vars, n_vars)),
                       ParamsMatrix{T}(S_inds, S_consts, (n_vars, n_vars)),
                       sparse(1:n_observed, F_inds, ones(T, length(F_inds)), n_observed, n_vars),
                       !isnothing(M_inds) ? ParamsVector{T}(M_inds, M_consts, (n_vars,)) : nothing,
                       params, colnames)
end

Base.convert(::Type{RAMMatrices}, partable::ParameterTable) = RAMMatrices(partable)

############################################################################################
### get parameter table from RAMMatrices
############################################################################################

function ParameterTable(ram::RAMMatrices;
                        params::Union{AbstractVector{Symbol}, Nothing} = nothing,
                        observed_var_prefix::Symbol = :obs,
                        latent_var_prefix::Symbol = :var)
    # defer parameter checks until we know which ones are used

    if !isnothing(ram.colnames)
        latent_vars = SEM.latent_vars(ram)
        observed_vars = SEM.observed_vars(ram)
        colnames = ram.colnames
    else
        observed_vars = [Symbol("$(observed_var_prefix)_$i") for i in 1:nobserved_vars(ram)]
        latent_vars = [Symbol("$(latent_var_prefix)_$i") for i in 1:nlatent_vars(ram)]
        colnames = vcat(observed_vars, latent_vars)
    end

    # construct an empty table
    partable = ParameterTable(observed_vars = observed_vars,
                              latent_vars = latent_vars,
                              params = isnothing(params) ? ram.params : params)

    # fill the table
    position_names = Dict{Int, Symbol}(1:length(colnames) .=> colnames)

    append_rows!(partable, ram.S, :S,
                 ram.params, position_names, skip_symmetric=true)
    append_rows!(partable, ram.A, :A,
                 ram.params, position_names)
    if !isnothing(ram.M)
        append_rows!(partable, ram.M, :M,
                     ram.params, position_names)
    end

    check_params(SEM.params(partable), partable.columns.param)

    return partable
end

Base.convert(::Type{<:ParameterTable}, ram::RAMMatrices;
             params::Union{AbstractVector{Symbol}, Nothing} = nothing) =
        ParameterTable(ram; params = params)

############################################################################################
### Pretty Printing
############################################################################################

function Base.show(io::IO, ram_matrices::RAMMatrices)
    print_type_name(io, ram_matrices)
    print_field_types(io, ram_matrices)
end

############################################################################################
### Additional Functions
############################################################################################

function matrix_to_relation(matrix::Symbol)
    if matrix == :A
        return :→
    elseif matrix == :S
        return :↔
    elseif matrix == :M
        return :→
    else
        throw(ArgumentError("Unsupported matrix $matrix, supported matrices are :A, :S and :M"))
    end
end

function partable_row(val, index, matrix::Symbol,
                      position_names::AbstractDict;
                      free::Bool = true)

    # variable names
    if matrix == :M
        from = Symbol("1")
        to = position_names[index]
    else
        from = position_names[index[2]]
        to = position_names[index[1]]
    end

    return (
        from = from,
        relation = matrix_to_relation(matrix),
        to = to,
        free = free,
        value_fixed = free ? 0.0 : val,
        start = 0.0,
        estimate = 0.0,
        param = free ? val : :const)
end

function append_rows!(partable::ParameterTable,
                      arr::ParamsArray, arr_name::Symbol,
                      params::AbstractVector,
                      position_names;
                      skip_symmetric::Bool = false)
    nparams(arr) == length(params) ||
        throw(ArgumentError("Length of parameters vector does not match the number of parameters in the matrix"))
    arr_ixs = eachindex(arr)

    # add parameters
    visited_indices = Set{eltype(arr_ixs)}()
    for (i, par) in enumerate(params)
        for j in param_occurences_range(arr, i)
            arr_ix = arr_ixs[arr.linear_indices[j]]
            skip_symmetric && (arr_ix ∈ visited_indices) && continue

            push!(partable, partable_row(par, arr_ix, arr_name, position_names, free=true))
            if skip_symmetric
                # mark index and its symmetric as visited
                push!(visited_indices, arr_ix)
                push!(visited_indices, CartesianIndex(arr_ix[2], arr_ix[1]))
            end
        end
    end

    # add constants
    for (i, _, val) in arr.constants
        arr_ix = arr_ixs[i]
        skip_symmetric && (arr_ix ∈ visited_indices) && continue
        push!(partable, partable_row(val, arr_ix, arr_name, position_names, free=false))
        if skip_symmetric
            # mark index and its symmetric as visited
            push!(visited_indices, arr_ix)
            push!(visited_indices, CartesianIndex(arr_ix[2], arr_ix[1]))
        end
    end

    return nothing
end

function Base.:(==)(mat1::RAMMatrices, mat2::RAMMatrices)
    res = ( (mat1.A == mat2.A) && (mat1.S == mat2.S) &&
            (mat1.F == mat2.F) && (mat1.M == mat2.M) &&
            (mat1.params == mat2.params) &&
            (mat1.colnames == mat2.colnames) )
    return res
end
