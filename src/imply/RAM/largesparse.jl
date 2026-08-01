############################################################################################
### Types
############################################################################################
@doc raw"""
Subtype of `SemImply` that implements the RAM notation for large-scale SEM model with
sparse upper triangular *A* (and sparse *(I - A)⁻¹*) and sparse block-diagonal *S*
(i.e. no covarations between latent and observed variables) matrices utilizing symbolic precomutation,
sparse Cholesky decomposition and sparse matrix multiplication.

# Constructor

    RAMLargeSparse(;specification,
        gradient = true,
        hessian = false,
        approximate_hessian = false,
        meanstructure = false,
        kwargs...)

# Arguments
- `specification`: either a `RAMMatrices` or `ParameterTable` object
- `meanstructure::Bool`: does the model have a meanstructure?
- `gradient::Bool`: is gradient-based optimization used
- `hessian::Bool`: is hessian-based optimization used
- `approximate_hessian::Bool`: for hessian based optimization: should the hessian be approximated
- `vech::Bool`: should the half-vectorization of Σ be computed (instead of the full matrix)
    (automatically set to true if any of the loss functions is SemWLS)

# Extended help

## Implementation
Subtype of `SemImply`.

## Interfaces
- `params(::RAMLargeSparse) `-> vector of parameter names
- `nparams(::RAMLargeSparse)` -> number of parameters

- `Σ(::RAMLargeSparse)` -> model implied covariance matrix
- `μ(::RAMLargeSparse)` -> model implied mean vector

Jacobians (only available in gradient! calls)
- `∇Σ(::RAMLargeSparse)` -> ``∂vec(Σ)/∂θᵀ``
- `∇μ(::RAMLargeSparse)` -> ``∂μ/∂θᵀ``

- `∇Σ_eval!(::RAMLargeSparse)` -> function to overwrite `∇Σ` in place,
    i.e. `∇Σ_eval!(∇Σ, θ)`. Normally, you do not want to use this but simply
    query `∇Σ(::RAMLargeSparse)`.

Hessians
The computation of hessians is more involved, and uses the "chain rule for
hessian matrices".
Therefore, we desribe it at length in the mathematical appendix of the online documentation,
and the relevant interfaces are omitted here.

Additional interfaces
- `has_meanstructure(::RAMLargeSparse)` -> `Val{Bool}` does the model have a meanstructure?

## RAM notation
The model implied covariance matrix is computed as
```math
    \Sigma = F(I-A)^{-1}S(I-A)^{-T}F^T
```
and for models with a meanstructure, the model implied means are computed as
```math
    \mu = F(I-A)^{-1}M
```
"""
mutable struct RAMLargeSparse{MS, SPEC, T, PM_S, PM_I_Aoo, PM_I_All,
                              F_I_A⁻¹oo, F_I_A⁻¹ll,
                              M_I_Aoo, M_I_All, M_Aol,
                              M_I_A⁻¹oo, M_I_A⁻¹ll, M_I_A⁻¹, M_FI_A, M_I_AooxS,
                              M_S, M_Σ, V_M, GM, GM_M, CHOL_Soo, CHOL_Sll} <: SemImply{MS,ExactHessian}
    ram::SPEC
    # parameterized S: [o,o] and [l,l] submatrices
    Soo_parr::PM_S
    Sll_parr::PM_S

    # parameterized I+A [o,o], [l,l] and [o,l] submatrices
    I_Aoo_parr::PM_I_Aoo # nothing if no observed-observed regression
    I_All_parr::PM_I_All # nothing if no latent-latent regression
    I_Aol_parr::PM_S

    # parameterized I+A [o,l] submatrix
    Aol_parr::PM_S

    # (I-A)⁻¹[o,o] and (I-A)⁻¹[l,l]⁻¹ evaluators
    I_A⁻¹oo_eval!::F_I_A⁻¹oo
    I_A⁻¹ll_eval!::F_I_A⁻¹ll

    # preallocated I-A submatrices
    I_Aoo::M_I_Aoo # I if no observed-observed regression
    I_All::M_I_All # I if no latent-latent regression
    I_Aol::M_Aol

    # preallocated I+A[o,l] submatrix
    Aol::M_Aol

    # preallocated (I-A)⁻¹ submatrices
    I_A⁻¹oo::M_I_A⁻¹oo # could be I
    I_A⁻¹ll::M_I_A⁻¹ll # could be I
    I_A⁻¹ol::M_Aol
    I_A⁻¹::M_I_A⁻¹

    # mappings from nonzero indices in (I-A)⁻¹ submatrices to
    # nonzero indices in the full (I-A)⁻¹
    I_A⁻¹oo_to_I_A⁻¹_destinds::Union{Vector{Int}, Nothing}
    I_A⁻¹ll_to_I_A⁻¹::Union{@NamedTuple{srcinds::Vector{Int}, destinds::Vector{Int}}, Nothing}
    I_A⁻¹ol_to_I_A⁻¹::@NamedTuple{srcinds::Vector{Int}, destinds::Vector{Int}}

    F⨉I_A⁻¹::M_FI_A # preallocated F⨉(I-A)⁻¹
    I_A⁻¹oo⨉Soo::M_I_AooxS
    I_A⁻¹ol⨉Sll::M_Aol

    S::Symmetric{T, M_S}

    # preallocated meanstructure
    M::V_M
    μ::V_M

    ∇A::GM
    ∇S::GM
    ∇M::GM_M

    # local (within-latent-block) indices of "active" latent vars,
    # i.e. those with a nonzero (free or fixed) entry in S.
    # Latents not in this set are "dependent" (deterministic) vars --
    # a linear combination of the active ones
    active_lat_ixs::Union{Vector{Int}, Nothing}
    # I_A⁻¹ol restricted to the active_lat_ixs columns (nothing if active_lat_ixs is nothing,
    # i.e. I_A⁻¹ol itself should be used directly); refreshed once per update!() call so that
    # the Σ⁻¹ Woodbury update does not need to re-slice I_A⁻¹ol on every access
    I_A⁻¹ol_active::Union{M_Aol, Nothing}

    Soo_shift::T
    Soo::Symmetric{T, M_S}
    _Soo_chol::CHOL_Soo
    Sll_shift::T
    Sll::Symmetric{T, M_Σ}
    _Sll_chol::CHOL_Sll
    _isposdef_S::Union{Bool, Nothing}
    _isposdef_Σ::Union{Bool, Nothing}

    _Σ_buf::M_Σ
    _Σ::Union{Symmetric{T, M_Σ}, Nothing}
    _Σ_chol::Union{Cholesky{T, M_Σ}, Nothing}

    _logdet_Σ::Union{T, Nothing}

    _Σ⁻¹oo_buf::M_Σ
    _Σ⁻¹lo_buf::M_Σ
    _Σ⁻¹lo_buf2::M_Σ
    _Σ⁻¹ll_buf::M_Σ
    _Σ⁻¹::Union{Symmetric{T, M_Σ}, Nothing}

    # options
    allow_indef_S::Bool
    try_fast_Σ::Bool

    # counters of Σ calculation roots taken
    n_fast::Int
    n_fast_failed::Int
    n_slow::Int
end

############################################################################################
### Constructors
############################################################################################

function RAMLargeSparse(spec::SemSpecification;
    gradient_required::Bool = true,
    simplify::Bool = true,
    Aⁿ_rewriter = nothing,
    max_An::Union{Nothing, Integer} = nothing,
    verbose::Bool = false,
    Sll_shift::Number = 0.0,
    Soo_shift::Number = Sll_shift,
    allow_indef_S::Bool = true,
    try_fast_Σ::Bool = true,
)
    ram = convert(RAMMatrices, spec)
    # sorting is required if one wants to omit F
    issorted(observed_var_indices(ram)) ||
        error("The order of observed variables in all variables array does not match their order in observed-only array. " *
              "RAMLargeSparse SEM implied model requires that they match")
    @assert issorted(latent_var_indices(ram))

    nobs = nobserved_vars(ram)
    nlat = nlatent_vars(ram)
    npars = nparams(ram)

    T = Float64
    randpars = randn(T, npars)

    sympars = (Symbolics.@variables θ[1:npars])[1]
    A_sym = sparse_materialize(Num, ram.A, sympars)
    A_par = materialize(Union{T, Symbol}, ram.A, params(ram))
    if istriu(A_sym)
        M_A = UpperTriangular
        M_I_A = UnitUpperTriangular
    elseif istril(A_par)
        M_A = LowerTriangular
        M_I_A = UnitLowerTriangular
    else
        error("RAMLargeSparse requires that A(θ) is upper or lower triangular")
    end
    verbose && @info "A matrix is $M_A"
    # I - A(θ) parametric array, in evaluate!() it is instantiated with -θ to get I - A(θ)
    I_A_par = copy(A_par)
    @inbounds for (i, v) in enumerate(I_A_par)
        isa(v, Number) && (v != 0) && (I_A_par[i] = -v)
    end
    @inbounds for i in diagind(I_A_par)
        @assert iszero(I_A_par[i])
        I_A_par[i] = one(T)
    end

    observed_ixs = observed_var_indices(ram)
    latent_ixs = latent_var_indices(ram)
    I_Aoo_par = I_A_par[observed_ixs, observed_ixs]
    Aoo_par = A_par[observed_ixs, observed_ixs]
    if any(!=(zero(T)), Aoo_par)
        I_Aoo_parr = ParamsArray{T}(I_Aoo_par, params(ram))
        I_Aoo_pre = M_I_A(sparse_materialize(I_Aoo_parr, randpars))
        @assert M_I_A == UnitLowerTriangular && istril(I_Aoo_pre) || M_I_A == UnitUpperTriangular && istriu(I_Aoo_pre)
    else
        verbose && @info "No observed-observed regressions detected, I - A[obs,obs] = I"
        I_Aoo_parr = nothing
        I_Aoo_pre = spdiagm(nobs, nobs, fill(one(T), nobs)) # I does not work with ldiv!(X, cholmod, I)
    end

    I_All_par = I_A_par[latent_ixs, latent_ixs]
    All_par = A_par[latent_ixs, latent_ixs]
    if any(!=(zero(T)), All_par)
        I_All_parr = ParamsArray{T}(I_All_par, params(ram))
        I_All_pre = M_I_A(materialize(I_All_parr, randpars))
        @assert M_I_A == UnitLowerTriangular && istril(I_All_pre) || M_I_A == UnitUpperTriangular && istriu(I_All_pre)
    else
        verbose && @info "No latent-latent regressions detected, I - A[lat,lat] = I"
        I_All_parr = nothing
        I_All_pre = I
    end

    Aol_parr = ParamsArray{T}(A_par[observed_ixs, latent_ixs], params(ram))
    Aol_pre = materialize(Aol_parr, randpars)
    I_Aol_parr = ParamsArray{T}(I_A_par[observed_ixs, latent_ixs], params(ram))
    I_Aol_pre = materialize(Aol_parr, randpars)

    verbose && @info "computing symbolic (I - A)⁻¹(θ) via Neumann series"
    I_A⁻¹_sym = neumann_series(A_sym; Aⁿ_rewriter, maxn = something(max_An, size(A_sym, 1)))
    @assert M_I_A == UnitUpperTriangular && istriu(I_A⁻¹_sym) || M_I_A == UnitLowerTriangular && istril(I_A⁻¹_sym)
    if simplify
        verbose && @info "  simplifying (I - A)⁻¹(θ)"
        # I_A⁻¹_sym = Symbolics.simplify.(I_A⁻¹_sym)
        Threads.@threads for i in eachindex(I_A⁻¹_sym.nzval)
            I_A⁻¹_sym.nzval[i] = Symbolics.simplify(I_A⁻¹_sym.nzval[i])
        end
    end
    if !isnothing(I_All_parr)
        verbose && @info "compiling in-place I_A⁻¹ll(θ) = (I - A(θ))⁻¹[lat, lat] = (I - A(θ)[lat, lat])⁻¹"
        _, I_A⁻¹ll_eval! = Symbolics.build_function(
            convert(Matrix, I_A⁻¹_sym[latent_ixs, latent_ixs]),
            sympars, expression=Val{false})
        verbose && @info "  generating initial I_A⁻¹ll(θ)..."
        I_A⁻¹ll_pre = M_I_A(zeros(T, nlat, nlat))
        verbose && @info "  $(nnz(parent(I_A⁻¹ll_pre))) nonzeros in I_A⁻¹[lat,lat]"
        I_A⁻¹ll_eval!(I_A⁻¹ll_pre, randpars)
    else # no latent-latent regression
        I_A⁻¹ll_eval! = nothing
        I_A⁻¹ll_pre = I
    end

    if !isnothing(I_Aoo_parr)
        verbose && @info "compiling in-place I_A⁻¹oo(θ) = (I - A(θ))⁻¹[obs, obs]"
        I_A⁻¹oo_sym = I_A⁻¹_sym[observed_ixs, observed_ixs]
        I_A⁻¹oo_eval, I_A⁻¹oo_eval! = Symbolics.build_function(I_A⁻¹oo_sym, sympars, expression=Val{false})
        verbose && @info "  generating initial I_A⁻¹o(θ)..."
        I_A⁻¹oo_pre = M_I_A(I_A⁻¹oo_eval(randpars))
        verbose && @info "  $(nnz(parent(I_A⁻¹oo_pre))) nonzeros in I_A⁻¹[obs,obs]"
    else
        I_A⁻¹oo_eval, I_A⁻¹oo_eval! = nothing, nothing
        I_A⁻¹oo_pre = I
    end

    # I_A⁻¹ol === Aol if no latent-latent and observed-observed regressions
    I_A⁻¹ol_pre = !isnothing(I_A⁻¹ll_eval!) || !isnothing(I_A⁻¹oo_eval!) ? similar(Aol_pre) : Aol_pre

    # materialize sparse I_A submatrices
    I_A⁻¹_pre = M_I_A(SparseMatrixCSC(size(I_A⁻¹_sym)..., I_A⁻¹_sym.colptr, I_A⁻¹_sym.rowval,
                      [isa(Symbolics.value(v), Number) ? T(Symbolics.value(v)) : rand(T) for v in I_A⁻¹_sym.nzval]))
    F⨉I_A⁻¹_pre = ram.F * I_A⁻¹_pre

    Σ_pre = Symmetric(zeros(T, nobs, nobs))
    S_par = materialize(ram.S, params(ram))
    S_pre = Symmetric(sparse_materialize(ram.S, randpars))
    Sol_par = S_par[observed_ixs, latent_ixs]
    Sol_nz = findall(!iszero, Sol_par)
    if !isempty(Sol_nz)
        error("S contains $(length(Sol_nz)) covariations between observed and latent variables, ",
              "RAMLargeSparse SEM implied model not applicable")
    end
    Soo_parr = ParamsArray{T}(S_par[observed_ixs, observed_ixs], params(ram))
    Sll_parr = ParamsArray{T}(S_par[latent_ixs, latent_ixs], params(ram))
    Soo_pre = Symmetric(sparse_materialize(Soo_parr, randpars))
    Soo_chol = SparseArrays.CHOLMOD.@cholmod_param final_ll = false begin
        SparseArrays.CHOLMOD.analyze(SparseArrays.CHOLMOD.Sparse(Soo_pre))
    end
    Sll_pre = Symmetric(materialize(Sll_parr, randpars))
    # detect "active" and "dependent" (deterministic) latent variables:
    # v is "dependent" if S[:, v] == 0 and S[v, :] == 0, so its effects are entirely determined
    # by the other (active) vars (v is a linear combination of active variables)
    active_lat_mask = falses(nlat)
    Sll_ixs = CartesianIndices((nlat, nlat))
    @inbounds for lin in Sll_parr.linear_indices
        ci = Sll_ixs[lin]
        active_lat_mask[ci[1]] = true
        active_lat_mask[ci[2]] = true
    end
    @inbounds for (lin, _, val) in Sll_parr.constants
        iszero(val) && continue
        ci = Sll_ixs[lin]
        active_lat_mask[ci[1]] = true
        active_lat_mask[ci[2]] = true
    end
    active_lat_ixs = findall(active_lat_mask)
    if length(active_lat_ixs) < nlat
        verbose && @info "  $(nlat - length(active_lat_ixs)) dependent (structural-zero-variance) latent variable(s) detected, excluded from Sll Cholesky"
        Sll_pre4chol = Symmetric(Sll_pre[active_lat_ixs, active_lat_ixs])
    else
        active_lat_ixs = nothing # no dependent latents
        Sll_pre4chol = Sll_pre
    end
    nactive_lat = isnothing(active_lat_ixs) ? nlat : length(active_lat_ixs)
    Sll_chol = cholesky(Sll_pre4chol, RowMaximum(); tol=1e-10, check=false)

    I_A⁻¹oo⨉Soo_pre = !isnothing(I_Aoo_parr) ? I_A⁻¹oo_pre * Soo_pre : I

    if gradient_required
        ∇A = sparse_gradient(T, ram.A)
        ∇S = sparse_gradient(T, ram.S)
    else
        ∇A = nothing
        ∇S = nothing
    end

    # μ
    if !isnothing(ram.M)
        MS = HasMeanStructure
        M_pre = materialize(ram.M, randpars)
        ∇M = gradient_required ? sparse_gradient(T, ram.M) : nothing
        μ_pre = zeros(T, nobs)
    else
        MS = NoMeanStructure
        M_pre = nothing
        ∇M = nothing
        μ_pre = nothing
    end

    return RAMLargeSparse{MS, typeof(ram), T, typeof(Soo_parr), typeof(I_Aoo_parr), typeof(I_All_parr),
                          typeof(I_A⁻¹oo_eval!), typeof(I_A⁻¹ll_eval!),
                          typeof(I_Aoo_pre), typeof(I_All_pre), typeof(Aol_pre),
                          typeof(I_A⁻¹oo_pre), typeof(I_A⁻¹ll_pre), typeof(I_A⁻¹_pre),
                          typeof(F⨉I_A⁻¹_pre), typeof(I_A⁻¹oo⨉Soo_pre),
                          typeof(parent(Soo_pre)), typeof(parent(Σ_pre)),
                          typeof(M_pre),
                          typeof(∇A), typeof(∇M),
                          typeof(Soo_chol), typeof(Sll_chol)}(
        ram, Soo_parr, Sll_parr, I_Aoo_parr, I_All_parr, I_Aol_parr, Aol_parr,
        I_A⁻¹oo_eval!, I_A⁻¹ll_eval!,
        I_Aoo_pre, I_All_pre, I_Aol_pre, Aol_pre,
        I_A⁻¹oo_pre, I_A⁻¹ll_pre, I_A⁻¹ol_pre, I_A⁻¹_pre,
        !isnothing(I_A⁻¹oo_eval!) ? nzsubmatrix_to_nzmatrix(
            parent(I_A⁻¹_pre), parent(I_A⁻¹oo_pre), observed_ixs, observed_ixs) : nothing,
        !isnothing(I_A⁻¹ll_eval!) ? begin
            srcinds, destinds = nzsubmatrix_to_nzmatrix(parent(I_A⁻¹_pre), parent(I_A⁻¹ll_pre),
                                                        latent_ixs, latent_ixs)
            (; srcinds, destinds)
        end : nothing,
        begin
            srcinds, destinds = nzsubmatrix_to_nzmatrix(
                parent(I_A⁻¹_pre), I_A⁻¹ol_pre, observed_ixs, latent_ixs)
            (; srcinds, destinds)
        end,
        F⨉I_A⁻¹_pre, I_A⁻¹oo⨉Soo_pre, similar(Aol_pre),
        S_pre, M_pre, μ_pre,
        ∇A, ∇S, ∇M,
        active_lat_ixs,
        isnothing(active_lat_ixs) ? nothing : zeros(T, nobs, length(active_lat_ixs)),
        Soo_shift, Soo_pre, Soo_chol,
        Sll_shift, Sll_pre, Sll_chol,
        nothing, nothing,
        Σ_pre, nothing, nothing, nothing,
        zeros(T, nobs, nobs),
        zeros(T, nactive_lat, nobs), zeros(T, nactive_lat, nobs),
        zeros(T, nactive_lat, nactive_lat),
        nothing,
        allow_indef_S, try_fast_Σ,
        0, 0 ,0
    )
end

iszeroAll(implied::RAMLargeSparse) = isnothing(implied.I_All_parr)
iszeroAoo(implied::RAMLargeSparse) = isnothing(implied.I_Aoo_parr)

############################################################################################
### objective, gradient, hessian
############################################################################################

function update!(targets::EvaluationTargets, implied::RAMLargeSparse, par)
    reset_covars!(implied)

    # update S
    materialize!(implied.S, implied.ram.S, par)

    # update (I - A)⁻¹
    materialize!(implied.Aol, implied.Aol_parr, par)
    if !isnothing(implied.I_A⁻¹oo_eval!) # only if I_A⁻¹o is not constant
        implied.I_A⁻¹oo_eval!(parent(implied.I_A⁻¹oo), par)
        @inbounds parent(implied.I_A⁻¹).nzval[implied.I_A⁻¹oo_to_I_A⁻¹_destinds] .= parent(implied.I_A⁻¹oo).nzval

        I_A⁻¹ol_1 = isnothing(implied.I_A⁻¹ll_eval!) ? implied.I_A⁻¹ol : implied.I_A⁻¹ol⨉Sll
        mul!(I_A⁻¹ol_1, implied.I_A⁻¹oo, implied.Aol)
    else
        I_A⁻¹ol_1 = implied.Aol
    end
    if !isnothing(implied.I_A⁻¹ll_eval!)
        implied.I_A⁻¹ll_eval!(parent(implied.I_A⁻¹ll), par)
        @inbounds parent(implied.I_A⁻¹).nzval[implied.I_A⁻¹ll_to_I_A⁻¹.destinds] .= parent(implied.I_A⁻¹ll)[implied.I_A⁻¹ll_to_I_A⁻¹.srcinds]
        mul!(implied.I_A⁻¹ol, I_A⁻¹ol_1, implied.I_A⁻¹ll)
    end # otherwise I_A⁻¹ol === I_A⁻¹ol_1 (=== Aol of both regressions are zero)
    @inbounds parent(implied.I_A⁻¹).nzval[implied.I_A⁻¹ol_to_I_A⁻¹.destinds] .= implied.I_A⁻¹ol[implied.I_A⁻¹ol_to_I_A⁻¹.srcinds]

    # dependent latents don't contribute to Σ (their S row/col is all-zero), so cache the
    # active-columns-only subset once here rather than re-slicing it in update_Σ⁻¹_sparse!
    isnothing(implied.active_lat_ixs) ||
        copyto!(implied.I_A⁻¹ol_active, @view implied.I_A⁻¹ol[:, implied.active_lat_ixs])

    # update F⨉I_A⁻¹
    mul!(implied.F⨉I_A⁻¹, implied.ram.F, implied.I_A⁻¹)

    # update μ
    if MeanStructure(implied) === HasMeanStructure
        materialize!(implied.M, implied.ram.M, par)
        mul!(implied.μ, implied.F⨉I_A⁻¹, implied.M)
    end

    # required only for Σ⁻¹ and logdet(Σ), but requires par, which is not passed to function update_Σ⁻¹!()
    materialize!(implied.Sll, implied.Sll_parr, par)
    materialize!(implied.Soo, implied.Soo_parr, par)

    # update (I - A)
    mpar = -par
    isnothing(implied.I_Aoo_parr) || materialize!(implied.I_Aoo, implied.I_Aoo_parr, mpar)
    isnothing(implied.I_All_parr) || materialize!(implied.I_All, implied.I_All_parr, mpar)
    materialize!(implied.I_Aol, implied.I_Aol_parr, mpar)
end

function reset_covars!(implied::RAMLargeSparse)
    implied._isposdef_S = nothing
    implied._isposdef_Σ = nothing
    implied._logdet_Σ = nothing
    implied._Σ = nothing
    implied._Σ_chol = nothing
    implied._Σ⁻¹ = nothing
end

# fix conversion error in SparseArrays.jl
function _isposdef(F::SparseArrays.CHOLMOD.Factor)
    if issuccess(F)
        s = unsafe_load(pointer(F))
        if s.is_ll == 1
            return true
        else
            # try conversion to LLt
            SparseArrays.CHOLMOD.change_factor!(F, true, Bool(s.is_super), true, Bool(s.is_monotonic))
            b = issuccess(F)
            # convert back
            SparseArrays.CHOLMOD.change_factor!(F, false, Bool(s.is_super), true, Bool(s.is_monotonic))
            return b
        end
    else
        return false
    end
end

# "in-place" inverse for pivoted Cholesky
function _inv!(res::AbstractMatrix, C::CholeskyPivoted{<:LinearAlgebra.BlasFloat,<:StridedMatrix})
    ipiv = invperm(C.piv)
    @inbounds copy!(res, view(LinearAlgebra.copytri!(LAPACK.potri!(C.uplo, copy(C.factors)), C.uplo, true), ipiv, ipiv))
end

# updates Cholesky decomposition of So and Sl
function update_S_chol!(implied::RAMLargeSparse)
    if isnothing(implied._isposdef_S)
        if !isnothing(implied.active_lat_ixs)
            s_idx = implied.active_lat_ixs
            Sll = copy!(parent(implied._Sll_chol.U), @view parent(implied.Sll)[s_idx, s_idx])
        else
            Sll = copy!(parent(implied._Sll_chol.U), implied.Sll)
        end
        if implied.Sll_shift != 0.0
            @inbounds for i in diagind(Sll)
                Sll[i] += implied.Sll_shift
            end
        end
        implied._Sll_chol = cholesky!(Sll, RowMaximum(); check=false)
        implied._isposdef_S = isposdef(implied._Sll_chol)
        if implied._isposdef_S # Sll (active sub-block) is posdef
            SparseArrays.CHOLMOD.cholesky!(implied._Soo_chol, implied.Soo; shift=implied.Soo_shift, check=false)
            implied._isposdef_S = _isposdef(implied._Soo_chol) # Soo Cholesky succeeded
        end
        # if !isposdef(implied._Sll_chol)
        #     # minimal eigen value of Sll
        #     Sll_eigvals = round.(eigvals(convert(Matrix{Float64}, parent(implied.Sll))), digits=4)
        #     @show Sll_eigvals[1:3]
        #     Soo_eigvals = round.(eigvals(convert(Matrix{Float64}, parent(implied.Soo))), digits=4)
        #     @show Soo_eigvals[1:3]
        #     Sll = copy(implied.Sll)
        #     Sll_dense = convert(Matrix{Float64}, Sll)
        #     Sll_chol = bunchkaufman(Sll_dense, check=false)
        #     @show Sll_chol.D
        #     copy!(parent(Sll).nzval, round.(parent(Sll).nzval, digits=3))
        #     let ctxt = IOContext(stdout, :compact => false)
        #         #print("Soo="); show(ctxt, "text/plain", implied.Soo); println()
        #         print("Sll="); show(ctxt, "text/plain", Sll); println()
        #     end
        # end
        #isposdef(implied._Soo_chol) || @info "Soo not posdef"
    end
end

# direct way of calculating Σoo cholesky from Σoo when S is not positive definite
function update_Σ_chol!(implied::RAMLargeSparse)
    if isnothing(implied._Σ_chol)
        BLAS.blascopy!(length(implied.Σ), parent(implied.Σ), 1, implied._Σ⁻¹oo_buf, 1)
        implied._Σ_chol = cholesky!(Symmetric(implied._Σ⁻¹oo_buf), check=false)
    end
end

function update_Σ!(implied::RAMLargeSparse)
    if isnothing(implied._Σ)
        # calculate Σ = F(I-A)⁻¹S(I-A)⁻ᵀFᵀ: first dense part (Sll)
        implied._Σ = Symmetric(X_A_Xt!(implied._Σ_buf, implied.Sll, implied.I_A⁻¹ol, X_A_buf = implied.I_A⁻¹ol⨉Sll))
        # Σ: sparse part (Soo)
        if implied.I_A⁻¹oo != I
            X_A_Xt!(parent(implied._Σ), implied.Soo, implied.I_A⁻¹oo, 1, 1, X_A_buf = implied.I_A⁻¹oo⨉Soo)
        else
            parent(implied._Σ) .+= implied.Soo
        end
    end
end

function isposdef_S(implied::RAMLargeSparse)
    isnothing(implied._isposdef_S) && update_S_chol!(implied)
    return implied._isposdef_S
end

function isposdef_Σ(implied::RAMLargeSparse)
    if isnothing(implied._isposdef_Σ)
        update_S_chol!(implied)
        if implied._isposdef_S || !implied.allow_indef_S
            implied._isposdef_Σ = implied._isposdef_S # Σoo is posdef since the whole big Σ is posdef
        else # slower path of Cholesky decomposition of the Σoo
            update_Σ_chol!(implied)
            implied._isposdef_Σ = isposdef(implied._Σ_chol)
        end
    end
    return implied._isposdef_Σ
end

# updates Σ⁻¹ and logdet(Σ)
function update_Σ⁻¹!(implied::RAMLargeSparse)
    isnothing(implied._logdet_Σ) || return nothing # skip if already updated
    update_S_chol!(implied)
    if implied.try_fast_Σ && isposdef_S(implied) # "sparse" (faster?) path
        implied.n_fast += 1
        update_Σ⁻¹_sparse!(implied)
        # update_Σ⁻¹_sparse!() may return without updating Σ⁻¹ if discovered numerical issues
        isnothing(implied._logdet_Σ) && (implied.n_fast_failed += 1)
    end
    if isnothing(implied._logdet_Σ)
        if (isposdef_S(implied) || implied.allow_indef_S) && isposdef_Σ(implied) # "dense" (slow) path
            implied.n_slow += 1
            update_Σ_chol!(implied)
            implied._logdet_Σ = logdet(implied._Σ_chol)
            implied._Σ⁻¹ = Symmetric(LinearAlgebra.inv!(implied._Σ_chol))
            implied._Σ_chol = nothing # invalidate Σ_chol since it got inverted
        else
            implied._logdet_Σ = NaN
            throw(LinearAlgebra.PosDefException(0))
        end
    end
end

function update_Σ⁻¹_sparse!(implied::RAMLargeSparse)
    isposdef_S(implied) || throw(LinearAlgebra.PosDefException(0))

    # Ω_oo = I_Aooᵀ Soo⁻¹ I_Aoo -- observed-observed block of the (I-A)ᵀS⁻¹(I-A) precision.
    # This does not involve any latent variables (dependent or stochastic), so it is
    # unaffected by the presence of dependent (structural-zero-variance) latents.
    Σ⁻¹oo = Symmetric(implied._Σ⁻¹oo_buf)
    if iszeroAoo(implied)
        # Σ⁻¹oo = Soo⁻¹, skip the multiplications by I_Aoo
        copy!(parent(Σ⁻¹oo), implied._Soo_chol \ implied.I_Aoo)
    else
        # parent() because CHOLMOD dispatch does not support triangular matrices:
        # the dispatch goes to generic triangular matrices
        Soo⁻½⨉I_Aoo = implied._Soo_chol.PtL \ parent(implied.I_Aoo)
        Xt_X!(parent(Σ⁻¹oo), Soo⁻½⨉I_Aoo)
    end

    # G_l = (I-A)⁻¹[obs, active-latent]: reuse the cached active-columns subset of I_A⁻¹ol
    # (dependent latents have all-zero S rows/cols, so they never contribute to Σ)
    G_l = isnothing(implied.active_lat_ixs) ? implied.I_A⁻¹ol : implied.I_A⁻¹ol_active
    # The work buffers are preallocated at the active latent size.
    ΩooG_lᵀ = implied._Σ⁻¹lo_buf # G_lᵀ Ω_oo, i.e. (Ω_oo G_l)ᵀ since Ω_oo is symmetric
    mul!(ΩooG_lᵀ, G_l', parent(Σ⁻¹oo))

    # inner = Sll_ss⁻¹ + G_lᵀ Ω_oo G_l, the Woodbury "capacitance" matrix
    # (size = number of active latents, always ≤ the full latent block)
    inner_buf = implied._Σ⁻¹ll_buf
    _inv!(inner_buf, implied._Sll_chol) # inner_buf = Sll_ss⁻¹
    mul!(inner_buf, ΩooG_lᵀ, G_l, true, true) # inner_buf += G_lᵀ Ω_oo G_l
    inner = Symmetric(inner_buf)
    if implied.Sll_shift != 0.0
        # add shift to the diagonal of inner
        @inbounds for i in diagind(inner)
            parent(inner)[i] += implied.Sll_shift
        end
    end
    inner_chol = cholesky!(inner; check=false)
    if !isposdef(inner_chol)
        # abort calculation, fallback to dense branch
        return
    end
    # by the matrix determinant lemma:
    # logdet(Σoo) = logdet(Soo) + logdet(Sll_ss) + logdet(inner)
    implied._logdet_Σ = logdet(implied._Soo_chol) + logdet(implied._Sll_chol) + logdet(inner_chol)

    # by the Woodbury identity:
    # Σoo⁻¹ = Ω_oo − Ω_oo G_l inner⁻¹ G_lᵀ Ω_oo
    # solve in place into a second k×nobs buffer (ΩooG_lᵀ is still needed below, so it
    # cannot be overwritten by the ldiv!) and accumulate the correction directly into
    # Σ⁻¹oo via a 5-arg mul!, avoiding an nobs×nobs temporary altogether
    inner⁻¹⨉ΩooG_lᵀ = implied._Σ⁻¹lo_buf2
    copyto!(inner⁻¹⨉ΩooG_lᵀ, ΩooG_lᵀ)
    ldiv!(inner_chol, inner⁻¹⨉ΩooG_lᵀ)
    mul!(parent(Σ⁻¹oo), ΩooG_lᵀ', inner⁻¹⨉ΩooG_lᵀ, -1, 1)
    implied._Σ⁻¹ = Σ⁻¹oo
end

Base.propertynames(implied::RAMLargeSparse, private::Bool=false) =
    (:logdet_Σ, :Σ, :Σ⁻¹, :ram_matrices,
     (private ? fieldnames(typeof(implied)) :
                filter(!startswith_undescore, fieldnames(typeof(implied))))...)

@inline function Base.getproperty(implied::RAMLargeSparse, name::Symbol)
    if name == :logdet_Σ # lazy logdet(Σ)
        isnothing(implied._logdet_Σ) && update_Σ⁻¹!(implied)
        return implied._logdet_Σ
    elseif name == :Σ # lazy Σ
        isnothing(implied._Σ) && update_Σ!(implied)
        return implied._Σ
    elseif name == :Σ⁻¹ # lazy Σ⁻¹
        isnothing(implied._Σ⁻¹) && update_Σ⁻¹!(implied)
        return implied._Σ⁻¹
    elseif name == :ram_matrices # fix the non-standard name for the RAM matrices
        return implied.ram
    else
        return getfield(implied, name)
    end
end
