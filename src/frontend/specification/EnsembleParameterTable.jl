############################################################################################
### Types
############################################################################################

mutable struct EnsembleParameterTable{C} <: AbstractParameterTable
    tables::C
end

############################################################################################
### Constructors
############################################################################################

# constuct an empty table
function EnsembleParameterTable(::Nothing)
    tables = Dict{Symbol, ParameterTable}()
    return EnsembleParameterTable(tables)
end

############################################################################################
### Convert to other types
############################################################################################

function Base.convert(::Type{Dict}, partable::EnsembleParameterTable)
    return convert(Dict, partable.tables)
end

function Base.convert(
    ::Type{Dict{K, RAMMatrices}},
    partables::EnsembleParameterTable;
    params::Union{AbstractVector{Symbol}, Nothing} = nothing,
) where {K}
    isnothing(params) || (params = SEM.params(partables))

    return Dict{K, RAMMatrices}(
        K(key) => RAMMatrices(partable; params = params) for
        (key, partable) in pairs(partables.tables)
    )
end

#= function DataFrame(
        partable::ParameterTable;
        columns = nothing)
    if isnothing(columns) columns = keys(partable.columns) end
    out = DataFrame([key => partable.columns[key] for key in columns])
    return DataFrame(out)
end =#

############################################################################################
### get parameter table from RAMMatrices
############################################################################################

function EnsembleParameterTable(args...; groups)
    partable = EnsembleParameterTable(nothing)

    for (group, ram_matrices) in zip(groups, args)
        push!(partable.tables, group => ParameterTable(ram_matrices))
    end

    return partable
end

############################################################################################
### Pretty Printing
############################################################################################

function Base.show(io::IO, partable::EnsembleParameterTable)
    print(io, "EnsembleParameterTable with groups: ")
    for key in keys(partable.tables)
        print(io, "|", key, "|")
    end
    print(io, "\n")
    for key in keys(partable.tables)
        print("\n")
        print(io, key, ": \n")
        print(io, partable.tables[key])
    end
end

############################################################################################
### Additional Methods
############################################################################################

# get the vector of all parameters in the table
# the position of the parameter is based on its first appearance in the table (and the ensemble)
function params(partable::EnsembleParameterTable)
    params = Vector{Symbol}()
    for tbl in values(partable.tables)
        append!(params, tbl.columns[:identifier])
    end
    return filter!(!=(:const), unique!(params)) # exclude constants
end


# Sorting ----------------------------------------------------------------------------------

function sort!(ensemble_partable::EnsembleParameterTable)
    for partable in values(ensemble_partable.tables)
        sort!(partable)
    end

    return ensemble_partable
end

function sort(partable::EnsembleParameterTable)
    new_partable = deepcopy(partable)
    sort!(new_partable)
    return new_partable
end

# add a row --------------------------------------------------------------------------------

# do we really need this?
function Base.push!(partable::EnsembleParameterTable, d::AbstractDict, group)
    push!(partable.tables[group], d)
end

Base.getindex(partable::EnsembleParameterTable, group) = partable.tables[group]

############################################################################################
### Update Partable from Fitted Model
############################################################################################

# update generic ---------------------------------------------------------------------------
function update_partable!(
    partable::EnsembleParameterTable,
    model_identifier::AbstractDict,
    vec,
    column,
)
    for k in keys(partable.tables)
        update_partable!(partable.tables[k], model_identifier, vec, column)
    end
    return partable
end
