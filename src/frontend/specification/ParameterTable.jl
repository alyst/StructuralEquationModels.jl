############################################################################################
### Types
############################################################################################

struct ParameterTable{C} <: AbstractParameterTable
    columns::C
    observed_vars::Vector{Symbol}
    latent_vars::Vector{Symbol}
    sorted_vars::Vector{Symbol}
end

############################################################################################
### Constructors
############################################################################################

# constuct an empty table
function ParameterTable(;
    observed_vars::Union{AbstractVector{Symbol}, Nothing} = nothing,
    latent_vars::Union{AbstractVector{Symbol}, Nothing} = nothing,
)
    columns = Dict{Symbol, Any}(
        :from => Vector{Symbol}(),
        :parameter_type => Vector{Symbol}(),
        :to => Vector{Symbol}(),
        :free => Vector{Bool}(),
        :value_fixed => Vector{Float64}(),
        :start => Vector{Float64}(),
        :estimate => Vector{Float64}(),
        :identifier => Vector{Symbol}(),
        :start => Vector{Float64}(),
    )

    return ParameterTable(columns,
        !isnothing(observed_vars) ? copy(observed_vars) : Vector{Symbol}(),
        !isnothing(latent_vars) ? copy(latent_vars) : Vector{Symbol}(),
        sorted_vars)
end

############################################################################################
### Convert to other types
############################################################################################

function Base.convert(::Type{Dict}, partable::ParameterTable)
    return partable.columns
end

function DataFrames.DataFrame(
    partable::ParameterTable;
    columns::Union{AbstractVector{Symbol}, Nothing} = nothing,
)
    if isnothing(columns)
        columns = [col for (col, vals) in pairs(partable.columns)]
    end
    return DataFrame([col => partable.columns[col] for col in columns])
end

############################################################################################
### Pretty Printing
############################################################################################

function Base.show(io::IO, partable::ParameterTable)
    relevant_columns = [
        :from,
        :parameter_type,
        :to,
        :free,
        :value_fixed,
        :start,
        :estimate,
        :se,
        :identifier,
    ]
    shown_columns = filter!(
        col -> haskey(partable.columns, col) && length(partable.columns[col]) > 0,
        relevant_columns,
    )

    as_matrix = mapreduce(col -> partable.columns[col], hcat, shown_columns)
    pretty_table(
        io,
        as_matrix,
        header = (shown_columns, [eltype(partable.columns[col]) for col in shown_columns]),
        tf = PrettyTables.tf_compact,
    )

    print(io, "Latent Variables:    $(partable.latent_vars) \n")
    print(io, "Observed Variables:  $(partable.observed_vars) \n")
end

############################################################################################
### Additional Methods
############################################################################################

# Iteration --------------------------------------------------------------------------------
ParameterTableRow = @NamedTuple begin
    from::Symbol
    parameter_type::Symbol
    to::Symbol
    free::Bool
    value_fixed::Any
    identifier::Symbol
end

Base.getindex(partable::ParameterTable, i::Integer) = (
    partable.columns[:from][i],
    partable.columns[:parameter_type][i],
    partable.columns[:to][i],
    partable.columns[:free][i],
    partable.columns[:value_fixed][i],
    partable.columns[:identifier][i],
)

Base.length(partable::ParameterTable) = length(first(values(partable.columns)))
Base.eachindex(partable::ParameterTable) = Base.OneTo(length(partable))

Base.eltype(::Type{<:ParameterTable}) = ParameterTableRow
Base.iterate(partable::ParameterTable) = iterate(partable, 1)
Base.iterate(partable::ParameterTable, i::Integer) =
    i > length(partable) ? nothing : (partable[i], i + 1)


# get the vector of all parameters in the table
# the position of the parameter is based on its first appearance in the table (and the ensemble)
params(partable::ParameterTable) =
    filter!(!=(:const), unique(partable.columns[:identifier]))


# Sorting ----------------------------------------------------------------------------------
struct CyclicModelError <: Exception
    msg::AbstractString
end

Base.showerror(io::IO, e::CyclicModelError) = print(io, e.msg)

function Base.sort!(partable::ParameterTable)
    vars = [
        partable.latent_vars
        partable.observed_vars
    ]

    is_regression = [
        (partype == :→) && (from != Symbol("1")) for
        (partype, from) in zip(partable.columns[:parameter_type], partable.columns[:from])
    ]

    to = partable.columns[:to][is_regression]
    from = partable.columns[:from][is_regression]

    sorted_vars = Vector{Symbol}()

    while !isempty(vars)
        acyclic = false

        for (i, var) in enumerate(vars)
            if !(var ∈ to)
                push!(sorted_vars, var)
                deleteat!(vars, i)
                delete_edges = from .!= var
                to = to[delete_edges]
                from = from[delete_edges]
                acyclic = true
            end
        end

        acyclic ||
            throw(CyclicModelError("your model is cyclic and therefore can not be ordered"))
    end

    copyto!(resize!(partable.sorted_vars, length(sorted_vars)), sorted_vars)

    return partable
end

function Base.sort(partable::ParameterTable)
    new_partable = deepcopy(partable)
    sort!(new_partable)
    return new_partable
end

# add a row --------------------------------------------------------------------------------

function Base.push!(partable::ParameterTable, d::Union{AbstractDict{Symbol}, NamedTuple})
    for (key, val) in pairs(d)
        push!(partable.columns[key], val)
    end
end

Base.push!(partable::ParameterTable, d::Nothing) = nothing

############################################################################################
### Update Partable from Fitted Model
############################################################################################

# update generic ---------------------------------------------------------------------------

function update_partable!(
    partable::ParameterTable,
    model_identifier::AbstractDict,
    values::AbstractVector,
    column::Symbol,
)
    coldata = get!(() -> Vector{eltype(values)}(), partable.columns, column)
    resize!(coldata, length(partable))
    for (i, id) in enumerate(partable.columns[:identifier])
        if !(id == :const)
            coldata[i] = values[model_identifier[id]]
        elseif id == :const
            coldata[i] = zero(eltype(values))
        end
    end
    return partable
end

"""
    update_partable!(partable::AbstractParameterTable, sem_fit::SemFit, values, column)

Write `vec` to `column` of `partable`.

# Arguments
- `vec::Vector`: has to be in the same order as the `model` parameters
"""
update_partable!(
    partable::AbstractParameterTable,
    sem_fit::SemFit,
    values::AbstractVector,
    column::Symbol,
) = update_partable!(partable, identifier(sem_fit), values, column)

# update estimates -------------------------------------------------------------------------
"""
    update_estimate!(
        partable::AbstractParameterTable,
        sem_fit::SemFit)

Write parameter estimates from `sem_fit` to the `:estimate` column of `partable`
"""
update_estimate!(partable::AbstractParameterTable, sem_fit::SemFit) =
    update_partable!(partable, sem_fit, sem_fit.solution, :estimate)

# update starting values -------------------------------------------------------------------
"""
    update_start!(partable::AbstractParameterTable, sem_fit::SemFit)
    update_start!(partable::AbstractParameterTable, model::AbstractSem, start_val; kwargs...)

Write starting values from `sem_fit` or `start_val` to the `:estimate` column of `partable`.

# Arguments
- `start_val`: either a vector of starting values or a function to compute starting values
    from `model`
- `kwargs...`: are passed to `start_val`
"""
update_start!(partable::AbstractParameterTable, sem_fit::SemFit) =
    update_partable!(partable, sem_fit, sem_fit.start_val, :start)

function update_start!(
    partable::AbstractParameterTable,
    model::AbstractSem,
    start_val;
    kwargs...,
)
    if !(start_val isa Vector)
        start_val = start_val(model; kwargs...)
    end
    return update_partable!(partable, identifier(model), start_val, :start)
end

# update partable standard errors ----------------------------------------------------------
"""
    update_se_hessian!(
        partable::AbstractParameterTable,
        sem_fit::SemFit;
        hessian = :finitediff)

Write hessian standard errors computed for `sem_fit` to the `:se` column of `partable`

# Arguments
- `hessian::Symbol`: how to compute the hessian, see [se_hessian](@ref) for more information.

# Examples

"""
function update_se_hessian!(
    partable::AbstractParameterTable,
    sem_fit::SemFit;
    hessian = :finitediff,
)
    se = se_hessian(sem_fit; hessian = hessian)
    return update_partable!(partable, sem_fit, se, :se)
end
