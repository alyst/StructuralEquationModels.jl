# convert data to a matrix of numbers
# with the columns matching the order of observed_vars
# (if observed_vars not specified, the column order is taken from the data)
# returns the matrix of data and the corresponding column names
function prepare_data(data::AbstractDataFrame,
                      observed_cols::Union{AbstractVector, Nothing},
                      spec::Union{SemSpecification, Nothing}
)
    obs_vars = nothing
    if !isnothing(observed_cols)
        obs_vars = Symbol.(observed_cols)
        data = data[:, observed_cols]
    elseif !isnothing(spec)
        obs_vars = observed_vars(spec)
        if !isnothing(obs_vars)
            data = data[:, obs_vars]
        end
    end
    if isnothing(obs_vars) # default symbol names
        obs_vars = Symbol.(names(data))
    end
    return Matrix(data), obs_vars
end

function prepare_data(data::AbstractMatrix,
                      observed_cols::Union{AbstractVector, Nothing},
                      spec::Union{SemSpecification, Nothing}
)
    obs_vars = nothing
    if !isnothing(observed_cols)
        obs_vars = Symbol.(observed_cols)
    elseif !isnothing(spec)
        obs_vars = observed_vars(spec)
    end
    if isnothing(obs_vars) # default symbol names
        obs_vars = [Symbol(i) for i in 1:size(data, 2)]
    end
    return data, obs_vars
end
