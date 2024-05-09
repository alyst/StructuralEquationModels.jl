# convert data to a matrix of numbers
# with the columns matching the order of observed_vars
# (if observed_vars not specified, the column order is taken from the data)
# returns the matrix of data and the corresponding column names
function prepare_data(data::AbstractDataFrame,
                      observed_vars::Union{AbstractVector, Nothing})
    if isnothing(observed_vars)
        obs_colnames = Symbol.(names(data))
    else
        obs_colnames = Symbol.(observed_vars)
        data = data[:, obs_colnames]
    end

    return Matrix(data), obs_colnames
end
