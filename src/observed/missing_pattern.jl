# data associated with the specific pattern of missing manifested variables
struct SemObservedMissingPattern{T,S}
    obs_mask::BitVector     # observed vars mask
    miss_mask::BitVector    # missing vars mask
    nobserved::Int
    nmissed::Int
    rows::Vector{Int}       # rows in original data
    data::Matrix{T}         # non-missing submatrix of data (vars × observations)

    obs_mean::Vector{S} # means of observed vars
    obs_cov::Symmetric{S, Matrix{S}}  # covariance of observed vars
end

function SemObservedMissingPattern(
    obs_mask::BitVector,
    rows::AbstractVector{<:Integer},
    data::AbstractMatrix
)
    T = nonmissingtype(eltype(data))

    pat_data = convert(Matrix{T}, view(data, rows, obs_mask))
    if size(pat_data, 1) > 1
        pat_mean, pat_cov = mean_and_cov(pat_data, 1, corrected=false)
        @assert size(pat_cov) == (size(pat_data, 2), size(pat_data, 2))
    else
        pat_mean = reshape(pat_data[1, :], 1, :)
        pat_cov = fill(zero(T), length(pat_mean), length(pat_mean))
    end

    miss_mask = .!obs_mask

    return SemObservedMissingPattern{T, eltype(pat_mean)}(
        obs_mask, miss_mask,
        sum(obs_mask), sum(miss_mask),
        rows, permutedims(pat_data),
        dropdims(pat_mean, dims=1), Symmetric(pat_cov))
end

n_man(pat::SemObservedMissingPattern) = length(pat.obs_mask)
n_obs(pat::SemObservedMissingPattern) = length(pat.rows)

nobserved_vars(pat::SemObservedMissingPattern) = pat.nobserved
nmissed_vars(pat::SemObservedMissingPattern) = pat.nmissed

function reorder_observed_vars!(pat::SemObservedMissingPattern, source_to_dest::AbstractVector{<:Integer})
    obs_dest = sort!(unique!(source_to_dest[pat.obs_mask])) # indices of observed vars after reordering
    obs_src2dest = [searchsortedfirst(obs_dest, dest)
                    for (src, dest) in enumerate(source_to_dest) if pat.obs_mask[src]]
    copy!(pat.obs_mask, pat.obs_mask[source_to_dest])
    copy!(pat.miss_mask, pat.miss_mask[source_to_dest])
    copy!(pat.data, pat.data[obs_src2dest, :])
    copy!(pat.obs_mean, pat.obs_mean[obs_src2dest])
    copy!(parent(pat.obs_cov), pat.obs_cov[obs_src2dest, obs_src2dest])
end
