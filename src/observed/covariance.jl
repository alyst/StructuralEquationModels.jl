"""
Type alias for [SemObservedData](@ref) with no data, but with mean and covariance.
"""
const SemObservedCovariance = SemObservedData{Nothing}

function SemObservedCovariance(;
        obs_cov::AbstractMatrix,
        obs_mean::Union{AbstractVector, Nothing} = nothing,
        n_obs::Integer,
        obs_colnames::AbstractVector{Symbol},
        kwargs...)

    nvars = size(obs_cov, 1)
    size(obs_cov, 2) == nvars || throw(DimensionMismatch("The covariance matrix should be square, $(size(obs_cov)) was found."))
    if isnothing(obs_mean)
        obs_mean = zeros(nvars)
    else
        length(obs_mean) == nvars || throw(DimensionMismatch("The length of the mean vector $(length(obs_mean)) does not match the size of the covariance matrix $(size(obs_cov))"))
    end
    length(obs_colnames) == nvars || throw(DimensionMismatch("The number of column names $(length(obs_colnames)) does not match the size of the covariance matrix $(size(obs_cov))"))

    return SemObservedData(nothing, obs_colnames, Symmetric(obs_cov), obs_mean, n_obs)
end
