"""
Type alias for [SemObservedData](@ref) with no data, but with mean and covariance.
"""
const SemObservedCovariance = SemObservedData{Nothing}

function SemObservedCovariance(
    obs_cov::AbstractMatrix,
    obs_mean::Union{AbstractVector, Nothing} = nothing;
    n_obs::Integer,
    specification = nothing,
    obs_colnames::Union{AbstractVector, Nothing} = nothing,
    kwargs...
)
    nvars = size(obs_cov, 1)
    size(obs_cov, 2) == nvars || throw(DimensionMismatch("The covariance matrix should be square, $(size(obs_cov)) was found."))

    if isnothing(obs_mean)
        obs_mean = zeros(nvars)
    else
        length(obs_mean) == nvars || throw(DimensionMismatch("The length of the mean vector $(length(obs_mean)) does not match the size of the covariance matrix $(size(obs_cov))"))
    end

    if !isnothing(obs_colnames)
        obs_vars = Symbol.(obs_colnames)
    elseif !isnothing(specification)
        obs_vars = observed_vars(specification)
    else
        obs_vars = [Symbol(i) for i in 1:nvars]
    end
    length(obs_vars) == nvars || throw(DimensionMismatch("The number of observed variables $(length(obs_vars)) does not match the size of the covariance matrix $(size(obs_cov))"))
    return SemObservedData(nothing, obs_vars, obs_cov, obs_mean, n_obs)
end
