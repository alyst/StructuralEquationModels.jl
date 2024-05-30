"""
For observed data without missings.

# Constructor

    SemObservedData(;
        specification,
        data,
        meanstructure = false,
        obs_colnames = nothing,
        kwargs...)

# Arguments
- `specification`: either a `RAMMatrices` or `ParameterTable` object (1)
- `data`: observed data
- `meanstructure::Bool`: does the model have a meanstructure?
- `obs_colnames::Vector{Symbol}`: column names of the data (if the object passed as data does not have column names, i.e. is not a data frame)

# Extended help
## Interfaces
- `n_obs(::SemObservedData)` -> number of observed data points
- `n_man(::SemObservedData)` -> number of manifest variables

- `get_data(::SemObservedData)` -> observed data
- `obs_cov(::SemObservedData)` -> observed.obs_cov
- `obs_mean(::SemObservedData)` -> observed.obs_mean

## Implementation
Subtype of `SemObserved`

## Remarks
(1) the `specification` argument can also be `nothing`, but this turns of checking whether
the observed data/covariance columns are in the correct order! As a result, you should only
use this if you are sure your observed data is in the right format.

## Additional keyword arguments:
- `spec_colnames::Vector{Symbol} = nothing`: overwrites column names of the specification object
- `compute_covariance::Bool ) = true`: should the covariance of `data` be computed and stored?
"""
struct SemObservedData{D <: Union{Nothing, AbstractMatrix}} <: SemObserved
    data::D
    observed_vars::Vector{Symbol}

    obs_cov::Matrix{Float64}
    obs_mean::Vector{Float64}
    n_obs::Int
end

# error checks
function check_arguments_SemObservedData(kwargs...)
    # data is a data frame,

end


function SemObservedData(data;
        specification = nothing,
        obs_colnames::Union{AbstractVector, Nothing} = nothing,
        kwargs...
)
    data, obs_vars = prepare_data(data, obs_colnames, specification)
    length(obs_vars) == size(data, 2) ||
        throw(DimensionMismatch("The number of observed variables ($(length(obs_vars))) " *
                                "does not match the number of columns in the data ($(size(data, 2)))"))
    obs_mean, obs_cov = mean_and_cov(data, 1)

    return SemObservedData(data, obs_vars,
        obs_cov, vec(obs_mean),
        size(data, 1))
end

############################################################################################
### Recommended methods
############################################################################################

n_obs(observed::SemObservedData) = observed.n_obs
n_man(observed::SemObservedData) = length(observed.observed_vars)

############################################################################################
### additional methods
############################################################################################

obs_cov(observed::SemObservedData) = observed.obs_cov
obs_mean(observed::SemObservedData) = observed.obs_mean

############################################################################################
### Additional functions
############################################################################################

# permutation that subsets and reorders source to matches the destination order ------------
function source_to_dest_perm(src::AbstractVector, dest::AbstractVector)
    if dest == src # exact match
        return eachindex(dest)
    else
        src_inds = Dict(el => i for (i, el) in enumerate(src))
        return [src_inds[el] for el in dest]
    end
end