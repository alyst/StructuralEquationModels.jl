############################################################################################
### Constants
############################################################################################

struct RAMConstant
    matrix::Symbol
    index::Union{Int, CartesianIndex{2}}
    value::Any
end

function Base.:(==)(c1::RAMConstant, c2::RAMConstant)
    res = ((c1.matrix == c2.matrix) && (c1.index == c2.index) && (c1.value == c2.value))
    return res
end

function append_RAMConstants!(
    constants::AbstractVector{RAMConstant},
    mtx_name::Symbol,
    mtx::AbstractArray,
)
    for (index, val) in pairs(mtx)
        if isa(val, Number) && !iszero(val)
            push!(constants, RAMConstant(mtx_name, index, val))
        end
    end
    return constants
end

function set_RAMConstant!(A, S, M, rc::RAMConstant)
    if rc.matrix == :A
        A[rc.index] = rc.value
    elseif rc.matrix == :S
        S[rc.index] = rc.value
        S[rc.index[2], rc.index[1]] = rc.value # symmetric
    elseif rc.matrix == :M
        M[rc.index] = rc.value
    end
end

function set_RAMConstants!(A, S, M, rc_vec::Vector{RAMConstant})
    for rc in rc_vec
        set_RAMConstant!(A, S, M, rc)
    end
end

############################################################################################
### Type
############################################################################################

# map from parameter index to linear indices of matrix/vector positions where it occurs
AbstractArrayParamsMap = AbstractVector{<:AbstractVector{<:Integer}}
ArrayParamsMap = Vector{Vector{Int}}

struct RAMMatrices <: SemSpecification
    A_ind::ArrayParamsMap
    S_ind::ArrayParamsMap
    F_ind::Vector{Int}
    M_ind::Union{ArrayParamsMap, Nothing}
    params::Vector{Symbol}
    colnames::Union{Vector{Symbol}, Nothing}
    constants::Vector{RAMConstant}
    size_F::Tuple{Int, Int}
end

############################################################################################
### Constructor
############################################################################################

function RAMMatrices(; A, S, F, M = nothing, params, colnames)
    A_indices = array_params_map(params, A)
    S_indices = array_params_map(params, S)
    M_indices = !isnothing(M) ? array_params_map(params, M) : nothing
    F_indices = [i for (i, col) in zip(axes(F, 2), eachcol(F)) if any(isone, col)]
    constants = Vector{RAMConstant}()
    append_RAMConstants!(constants, :A, A)
    append_RAMConstants!(constants, :S, S)
    isnothing(M) || append_RAMConstants!(constants, :M, M)
    return RAMMatrices(
        A_indices,
        S_indices,
        F_indices,
        M_indices,
        params,
        colnames,
        constants,
        size(F),
    )
end

############################################################################################
### get RAMMatrices from parameter table
############################################################################################

function RAMMatrices(partable::ParameterTable; par_id = nothing)
    if isnothing(par_id)
        params, n_par, par_positions = get_par_npar_indices(partable)
    else
        params, n_par, par_positions =
            par_id[:params], par_id[:n_par], par_id[:par_positions]
    end

    n_observed = length(partable.observed_vars)
    n_latent = length(partable.latent_vars)
    n_node = n_observed + n_latent

    # F indices
    F_ind =
        length(partable.sorted_vars) != 0 ?
        findall(∈(Set(partable.observed_vars)), partable.sorted_vars) :
        1:n_observed

    # indices of the colnames
    colnames =
        length(partable.sorted_vars) != 0 ? copy(partable.sorted_vars) :
        [
            partable.observed_vars
            partable.latent_vars
        ]
    col_indices = Dict(col => i for (i, col) in enumerate(colnames))

    # fill Matrices
    # known_labels = Dict{Symbol, Int64}()

    A_ind = Vector{Vector{Int64}}(undef, n_par)
    for i in 1:length(A_ind)
        A_ind[i] = Vector{Int64}()
    end
    S_ind = Vector{Vector{Int64}}(undef, n_par)
    S_ind .= [Vector{Int64}()]
    for i in 1:length(S_ind)
        S_ind[i] = Vector{Int64}()
    end

    # is there a meanstructure?
    M_ind =
        any(==(Symbol("1")), partable.columns[:from]) ? [Vector{Int64}() for _ in 1:n_par] :
        nothing

    # handle constants
    constants = Vector{RAMConstant}()

    for i in 1:length(partable)
        from, parameter_type, to, free, value_fixed, param = partable[i]

        row_ind = col_indices[to]
        col_ind = from != Symbol("1") ? col_indices[from] : nothing

        if !free
            if (parameter_type == :→) && (from == Symbol("1"))
                push!(constants, RAMConstant(:M, row_ind, value_fixed))
            elseif (parameter_type == :→)
                push!(
                    constants,
                    RAMConstant(:A, CartesianIndex(row_ind, col_ind), value_fixed),
                )
            elseif (parameter_type == :↔)
                push!(
                    constants,
                    RAMConstant(:S, CartesianIndex(row_ind, col_ind), value_fixed),
                )
            else
                error("Unsupported parameter type: $(parameter_type)")
            end
        else
            par_ind = par_positions[param]
            if (parameter_type == :→) && (from == Symbol("1"))
                push!(M_ind[par_ind], row_ind)
            elseif parameter_type == :→
                push!(A_ind[par_ind], row_ind + (col_ind - 1) * n_node)
            elseif parameter_type == :↔
                push!(S_ind[par_ind], row_ind + (col_ind - 1) * n_node)
                if row_ind != col_ind
                    push!(S_ind[par_ind], col_ind + (row_ind - 1) * n_node)
                end
            else
                error("Unsupported parameter type: $(parameter_type)")
            end
        end
    end

    return RAMMatrices(
        A_ind,
        S_ind,
        F_ind,
        M_ind,
        params,
        colnames,
        constants,
        (n_observed, n_node),
    )
end

Base.convert(::Type{RAMMatrices}, partable::ParameterTable) = RAMMatrices(partable)

############################################################################################
### get parameter table from RAMMatrices
############################################################################################

function ParameterTable(ram_matrices::RAMMatrices)
    colnames = ram_matrices.colnames

    partable = ParameterTable(
        observed_vars = colnames[ram_matrices.F_ind],
        latent_vars = colnames[setdiff(eachindex(colnames), ram_matrices.F_ind)],
    )

    # constants
    for c in ram_matrices.constants
        push!(partable, partable_row(c, colnames))
    end

    # parameters
    for (i, par) in enumerate(ram_matrices.params)
        append_partable_rows!(
            partable,
            colnames,
            par,
            i,
            ram_matrices.A_ind,
            ram_matrices.S_ind,
            ram_matrices.M_ind,
            ram_matrices.size_F[2],
        )
    end

    return partable
end

Base.convert(::Type{<:ParameterTable}, ram_matrices::RAMMatrices) =
    ParameterTable(ram_matrices)

############################################################################################
### get RAMMatrices from EnsembleParameterTable
############################################################################################

function RAMMatrices(partable::EnsembleParameterTable)
    ram_matrices = Dict{Symbol, RAMMatrices}()

    params, n_par, par_positions = get_par_npar_indices(partable)
    par_id =
        Dict(:params => params, :n_par => n_par, :par_positions => par_positions)

    for key in keys(partable.tables)
        ram_mat = RAMMatrices(partable.tables[key]; par_id = par_id)
        push!(ram_matrices, key => ram_mat)
    end

    return ram_matrices
end

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

function get_par_npar_indices(partable::ParameterTable)
    params = unique(partable.columns[:param])
    filter!(x -> x != :const, params)
    n_par = length(params)
    par_positions = Dict(params .=> 1:n_par)
    return params, n_par, par_positions
end

function get_par_npar_indices(partable::EnsembleParameterTable)
    params = Vector{Symbol}()
    for key in keys(partable.tables)
        append!(params, partable.tables[key].columns[:param])
    end
    params = unique(params)
    filter!(x -> x != :const, params)

    n_par = length(params)

    par_positions = Dict(params .=> 1:n_par)

    return params, n_par, par_positions
end

function matrix_to_parameter_type(matrix::Symbol)
    if matrix == :A
        return :→
    elseif matrix == :S
        return :↔
    elseif matrix == :M
        return :→
    else
        throw(
            ArgumentError(
                "Unsupported matrix $matrix, supported matrices are :A, :S and :M",
            ),
        )
    end
end

partable_row(c::RAMConstant, varnames::AbstractVector{Symbol}) = (
    from = varnames[c.index[2]],
    parameter_type = matrix_to_parameter_type(c.matrix),
    to = varnames[c.index[1]],
    free = false,
    value_fixed = c.value,
    start = 0.0,
    estimate = 0.0,
    param = :const,
)

function partable_row(
    par::Symbol,
    varnames::AbstractVector{Symbol},
    index::Integer,
    matrix::Symbol,
    n_nod::Integer,
)

    # variable names
    if matrix == :M
        from = Symbol("1")
        to = varnames[index]
    else
        cart_index = linear2cartesian(index, (n_nod, n_nod))

        from = varnames[cart_index[2]]
        to = varnames[cart_index[1]]
    end

    return (
        from = from,
        parameter_type = matrix_to_parameter_type(matrix),
        to = to,
        free = true,
        value_fixed = 0.0,
        start = 0.0,
        estimate = 0.0,
        param = par,
    )
end

function append_partable_rows!(
    partable::ParameterTable,
    varnames::AbstractVector{Symbol},
    par::Symbol,
    par_index::Integer,
    A_ind,
    S_ind,
    M_ind,
    n_nod::Integer,
)
    for ind in A_ind[par_index]
        push!(partable, partable_row(par, varnames, ind, :A, n_nod))
    end

    visited_S_indices = Set{Int}()
    for ind in S_ind[par_index]
        if ind ∉ visited_S_indices
            push!(partable, partable_row(par, varnames, ind, :S, n_nod))
            # mark index and its symmetric as visited
            push!(visited_S_indices, ind)
            cart_index = linear2cartesian(ind, (n_nod, n_nod))
            push!(
                visited_S_indices,
                cartesian2linear(
                    CartesianIndex(cart_index[2], cart_index[1]),
                    (n_nod, n_nod),
                ),
            )
        end
    end

    if !isnothing(M_ind)
        for ind in M_ind[par_index]
            push!(partable, partable_row(par, varnames, ind, :M, n_nod))
        end
    end

    return nothing
end

function Base.:(==)(mat1::RAMMatrices, mat2::RAMMatrices)
    res = (
        (mat1.A_ind == mat2.A_ind) &&
        (mat1.S_ind == mat2.S_ind) &&
        (mat1.F_ind == mat2.F_ind) &&
        (mat1.M_ind == mat2.M_ind) &&
        (mat1.params == mat2.params) &&
        (mat1.colnames == mat2.colnames) &&
        (mat1.size_F == mat2.size_F) &&
        (mat1.constants == mat2.constants)
    )
    return res
end
