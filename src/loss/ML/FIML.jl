############################################################################################
### Types
############################################################################################

# state of SemFIML for a specific missing pattern (`SemObservedMissingPattern` type)
struct SemFIMLPattern{T}
    ∇ind::Vector{Int}   # indices of co-observed variable pairs
    Σ⁻¹::Matrix{T}      # preallocated inverse of implied cov
    logdet::Ref{T}      # logdet of implied cov
    μ_diff::Vector{T}   # implied mean difference
end

# allocate arrays for pattern FIML
function SemFIMLPattern(pat::SemObservedMissingPattern)
    nobserved = nobserved_vars(pat)
    nmissed = nmissed_vars(pat)

    # linear indicies of co-observed variable pairs for each pattern
    Σ_linind = LinearIndices((n_man(pat), n_man(pat)))
    ∇ind = vec([Σ_linind[CartesianIndex(x, y)]
               for x in findall(pat.obs_mask), y in findall(pat.obs_mask)])

    return SemFIMLPattern(∇ind,
                          zeros(nobserved, nobserved),
                          Ref(NaN), zeros(nobserved))
end

function prepare!(fiml::SemFIMLPattern, pat::SemObservedMissingPattern, implied::SemImply)
    Σ = implied.Σ
    μ = implied.μ
    @inbounds @. @views begin
    fiml.Σ⁻¹ = Σ[pat.obs_mask, pat.obs_mask]
    fiml.μ_diff = pat.obs_mean - μ[pat.obs_mask]
    end
    Σ_chol = cholesky!(Symmetric(fiml.Σ⁻¹))
    fiml.logdet[] = logdet(Σ_chol)
    LinearAlgebra.inv!(Σ_chol) # updates fiml.Σ⁻¹
    #batch_sym_inv_update!(fiml, model)
    return fiml
end

function objective(fiml::SemFIMLPattern{T}, pat::SemObservedMissingPattern) where T
    F = fiml.logdet[] + dot(fiml.μ_diff, fiml.Σ⁻¹, fiml.μ_diff)
    if n_obs(pat) > 1
        F += dot(pat.obs_cov, fiml.Σ⁻¹)
        F *= n_obs(pat)
    end
    return F
end

function gradient!(JΣ, Jμ, fiml::SemFIMLPattern, pat::SemObservedMissingPattern)
    Σ⁻¹ = Symmetric(fiml.Σ⁻¹)
    μ_diff⨉Σ⁻¹ = fiml.μ_diff' * Σ⁻¹
    if n_obs(pat) > 1
        JΣ_pat = Σ⁻¹ * (I - pat.obs_cov * Σ⁻¹ - fiml.μ_diff * μ_diff⨉Σ⁻¹)
        JΣ_pat .*= n_obs(pat)
    else
        JΣ_pat = Σ⁻¹ * (I - fiml.μ_diff * μ_diff⨉Σ⁻¹)
    end
    @inbounds vec(JΣ)[fiml.∇ind] .+= vec(JΣ_pat)

    lmul!(2*n_obs(pat), μ_diff⨉Σ⁻¹)
    @inbounds Jμ[pat.obs_mask] .+= μ_diff⨉Σ⁻¹'
    return nothing
end

"""
Full information maximum likelihood estimation. Can handle observed data with missings.

# Constructor

    SemFIML(observed::SemObservedMissing, imply::SemImply)

# Arguments
- `observed::SemObservedMissing`: the observed part of the model
- `imply::SemImply`: [`SemImply`](@ref) instance

# Examples
```julia
my_fiml = SemFIML(my_observed, my_implied)
```

# Interfaces
Analytic gradients are available.

# Extended help
## Implementation
Subtype of `SemLossFunction`.
"""
struct SemFIML{O, I, T, W} <: SemLoss{O, I, ExactHessian}
    observed::O
    imply::I

    patterns::Vector{SemFIMLPattern{T}}

    imp_inv::Matrix{T}  # implied inverse

    commutator::CommutationMatrix
    #Q::SparseMatrixCSC{T}
    #q_indices::Vector{Int}
    #Q_nzixs2::Vector{Int}
    #q_indices2::Vector{Int}

    interaction::W
end

############################################################################################
### Constructors
############################################################################################

function SemFIML(observed::SemObservedMissing, imply::SemImply)
    # check integrity
    check_observed_vars(observed, imply)
#=
    # prepare sparse matrix Q
    n = nvars(specification)
    comm_indices = transpose_linear_indices(n)
    Qq_indices = Vector{Tuple{Int, Int}}()
    q_lin = LinearIndices((n, n))
    Q_lin = LinearIndices((n^2, n^2))
    for ij in CartesianIndices((n, n))
        i_n = (ij[1]-1)*n
        j_n = (ij[2]-1)*n
        for k in 1:n
            push!(Qq_indices, (Q_lin[i_n + k, j_n + k], q_lin[ij]))
            push!(Qq_indices, (Q_lin[comm_indices[i_n + k], j_n + k], q_lin[ij]))
        end
    end
    sort!(Qq_indices, by=first)

    Q_rowvals = Vector{Int}()
    Q_colptr = Vector{Int}()
    q_indices = Vector{Int}()
    Q_nzixs2 = Vector{Int}()
    q_indices2 = Vector{Int}()

    Q_cart = CartesianIndices((n^2, n^2))
    prev_Q_j = 0
    prev_Q_ij = 0
    for (Q_ij, q_kl) in Qq_indices
        if Q_ij != prev_Q_ij
            # assign Q element
            push!(q_indices, q_kl)
            Q_i, Q_j = Tuple(Q_cart[Q_ij])
            push!(Q_rowvals, Q_i)
            if Q_j > prev_Q_j
                push!(Q_colptr, length(Q_rowvals))
                prev_Q_j = Q_j
            end
            prev_Q_ij = Q_ij
        else
            # incrementing Q element (last one processed) with q
            push!(q_indices2, q_kl)
            push!(Q_nzixs2, length(Q_rowvals))
        end
    end
    push!(Q_colptr, length(Q_rowvals) + 1)
    @assert length(Q_colptr) == n^2 + 1
=#
    return SemFIML(observed, imply,
                   [SemFIMLPattern(pat) for pat in observed.patterns],
                   zeros(n_man(observed), n_man(observed)),
                   CommutationMatrix(nvars(imply)),
                   #SparseMatrixCSC(n^2, n^2, Q_colptr, Q_rowvals, ones(length(Q_rowvals))),
                   #q_indices, Q_nzixs2, q_indices2,
                   nothing)
end

############################################################################################
### methods
############################################################################################

function evaluate!(objective, gradient, hessian,
                   fiml::SemFIML, params)

    isnothing(hessian) || error("Hessian not implemented for FIML")

    copyto!(fiml.imp_inv, fiml.imply.Σ)
    Σ_chol = cholesky!(Symmetric(fiml.imp_inv); check = false)

    if !isposdef(Σ_chol)
        isnothing(objective) || (objective = non_posdef_return(params))
        isnothing(gradient) || fill!(gradient, 1)
        return objective
    end

    @inbounds for (pat_fiml, pat) in zip(fiml.patterns, fiml.observed.patterns)
        prepare!(pat_fiml, pat, fiml.imply)
    end

    scale = inv(n_obs(fiml.observed))
    isnothing(objective) || (objective = scale*F_FIML(eltype(params), fiml))
    isnothing(gradient) || (∇F_FIML!(gradient, fiml); gradient .*= scale)

    return objective
end

############################################################################################
### Recommended methods
############################################################################################

update_observed(lossfun::SemFIML, observed::SemObserved; kwargs...) =
    SemFIML(;observed = observed, kwargs...)

############################################################################################
### additional functions
############################################################################################

function ∇F_fiml_outer!(G, JΣ, Jμ, fiml::SemFIML{O, I}) where {O, I <: SemImplySymbolic}
    mul!(G, fiml.imply.∇Σ', JΣ) # should be transposed
    mul!(G, fiml.imply.∇μ', Jμ, -1, 1)
end

function ∇F_fiml_outer!(G, JΣ, Jμ, fiml::SemFIML)
    imply = fiml.imply

    P = kron(imply.F⨉I_A⁻¹, imply.F⨉I_A⁻¹)
    Iₙ = sparse(1.0I, size(imply.A)...)
    Q = kron(imply.S*imply.I_A⁻¹', Iₙ)
    Q .+= fiml.commutator * Q

    #= alternative way to calculate Q
    q = imply.S*imply.I_A⁻¹'
    Q = fiml.Q
    @inbounds Q.nzval .= q[fiml.q_indices]
    @inbounds Q.nzval[fiml.Q_nzixs2] .+= q[fiml.q_indices2]
    @assert Q ≈ Qalt
    =#

    ∇Σ = P*(imply.∇S + Q*imply.∇A)

    ∇μ = imply.F⨉I_A⁻¹*imply.∇M + kron((imply.I_A⁻¹*imply.M)', imply.F⨉I_A⁻¹)*imply.∇A

    mul!(G, ∇Σ', JΣ) # actually transposed
    mul!(G, ∇μ', Jμ, -1, 1)
end

function F_FIML(::Type{T}, fiml::SemFIML) where T
    F = zero(T)
    for (pat_fiml, pat) in zip(fiml.patterns, fiml.observed.patterns)
        F += objective(pat_fiml, pat)
    end
    return F
end

function ∇F_FIML!(G, fiml::SemFIML)
    Jμ = zeros(nobserved_vars(fiml))
    JΣ = zeros(nobserved_vars(fiml)^2)

    for (pat_fiml, pat) in zip(fiml.patterns, fiml.observed.patterns)
        gradient!(JΣ, Jμ, pat_fiml, pat)
    end
    ∇F_fiml_outer!(G, JΣ, Jμ, fiml)
end
