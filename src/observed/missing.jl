############################################################################################
### Types
############################################################################################

"""
For observed data with missing values.

# Constructor

    SemObservedMissing(;
        specification,
        data,
        obs_colnames = nothing,
        kwargs...)

# Arguments
- `specification`: either a `RAMMatrices` or `ParameterTable` object (1)
- `data`: observed data
- `obs_colnames::Vector{Symbol}`: column names of the data (if the object passed as data does not have column names, i.e. is not a data frame)

# Extended help
## Interfaces
- `n_obs(::SemObservedMissing)` -> number of observed data points
- `n_man(::SemObservedMissing)` -> number of manifest variables

- `get_data(::SemObservedMissing)` -> observed data
- `em_model(::SemObservedMissing)` -> `EmMVNModel` that contains the covariance matrix and mean vector found via expectation maximization

## Implementation
Subtype of `SemObserved`

## Remarks
(1) the `specification` argument can also be `nothing`, but this turns of checking whether
the observed data/covariance columns are in the correct order! As a result, you should only
use this if you are sure your observed data is in the right format.

## Additional keyword arguments:
- `spec_colnames::Vector{Symbol} = nothing`: overwrites column names of the specification object
"""
struct SemObservedMissing{T <: Real, S <: Real} <: SemObserved
    data::Matrix{Union{T, Missing}}
    observed_vars::Vector{Symbol}
    n_obs::Int
    patterns::Vector{SemObservedMissingPattern{T, S}}

    obs_cov::Matrix{S}
    obs_mean::Vector{S}
end

############################################################################################
### Constructors
############################################################################################

function SemObservedMissing(;
        data,
        obs_colnames = nothing,
        verbose::Bool = false,
        kwargs...)

    verbose && @info "Preparing data and observed variables..."
    data, observed_vars = prepare_data(data, obs_colnames)
    n_obs = size(data, 1)
    verbose && @info "  $(n_obs) observations of $(length(observed_vars)) variables"

    # detect all different missing patterns with their row indices
    verbose && @info "Detecting patterns of variables missing in observations..."
    pattern_to_rows = Dict{BitVector, Vector{Int}}()
    for (i, datarow) in zip(axes(data, 1), eachrow(data))
        pattern = BitVector(.!ismissing.(datarow))
        if sum(pattern) > 0 # skip all-missing rows
            pattern_rows = get!(() -> Vector{Int}(), pattern_to_rows, pattern)
            push!(pattern_rows, i)
        end
    end
    # process each pattern and sort from most to least number of observed vars
    patterns = [SemObservedMissingPattern(pat, rows, data)
                for (pat, rows) in pairs(pattern_to_rows)]
    sort!(patterns, by=nmissed_vars)
    verbose && @info "  $(length(patterns)) patterns detected"

    verbose && @info "Inferring N(μ, Σ) using EM algorithm..."
    em_cov, em_mean = em_mvn(patterns; verbose, kwargs...)

    return SemObservedMissing(data, observed_vars, n_obs, patterns, em_cov, em_mean)
end

n_obs(observed::SemObservedMissing) = observed.n_obs
n_man(observed::SemObservedMissing) = length(observed.observed_vars)

obs_cov(observed::SemObservedMissing) = observed.obs_cov
obs_mean(observed::SemObservedMissing) = observed.obs_mean
