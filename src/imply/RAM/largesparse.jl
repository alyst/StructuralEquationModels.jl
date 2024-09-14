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
mutable struct RAMLargeSparse{MS, SPEC, T, PM_S, F_I_Aoo, F_I_All, F_I_Aol, F_I_A⁻¹, M_F, M_A, M_I_A, M_S, M_Σ, GM, V_M, GM_M, CHOL_Soo, CHOL_Sll} <: SemImply{MS,ExactHessian}
    ram::SPEC
    # parameterized Soo and Sll matrices
    Soo_parr::PM_S
    Sll_parr::PM_S

    I_Aoo_eval!::F_I_Aoo
    I_All_eval!::F_I_All
    I_Aol_eval!::F_I_Aol
    I_A⁻¹_eval!::F_I_A⁻¹

    F::M_F
    M::V_M

    I_Aoo::M_I_A
    I_All::M_I_A
    I_Aol::M_A

    I_A⁻¹::M_I_A
    F⨉I_A⁻¹::M_A

    S::Symmetric{T, M_S}
    μ::V_M

    ∇A::GM
    ∇S::GM
    ∇M::GM_M

    Soo_shift::T
    Soo::Symmetric{T, M_S}
    _Soo_chol::CHOL_Soo
    Sll_shift::T
    Sll::Symmetric{T, M_S}
    _Sll_chol::CHOL_Sll
    allow_indef_S::Bool
    _isposdef_S::Union{Bool, Nothing}
    _isposdef_Σ::Union{Bool, Nothing}

    F⨉I_A⁻¹⨉S::M_Σ
    _Σ_buf::M_Σ
    _Σ::Union{Symmetric{T, M_Σ}, Nothing}
    _Σ_chol::Union{Cholesky{T, M_Σ}, Nothing}

    _logdet_Σ::Union{T, Nothing}

    _S⁻ʰ⨉I_Aoo_buf::M_A
    _Σ⁻¹oo_buf::M_Σ
    _Σ⁻¹lo_buf::M_Σ
    _Σ⁻¹lo_buf2::M_Σ
    _Σ⁻¹ll_buf::M_Σ
    _Σ⁻¹::Union{Symmetric{T, M_Σ}, Nothing}
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
    allow_indef_S::Bool = true
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
    pars = (Symbolics.@variables θ[1:npars])[1]

    F = copy(ram.F)

    verbose && @info "computing symbolic I - A⁻¹(θ)"
    T = Float64
    A_sym = sparse_materialize(Num, ram.A, pars)
    if istriu(A_sym)
        M_A = UpperTriangular
        M_I_A = UnitUpperTriangular
    elseif istril(A_sym)
        M_A = LowerTriangular
        M_I_A = UnitLowerTriangular
    else
        error("RAMLargeSparse requires that A(θ) is upper or lower triangular")
    end
    verbose && @info "A matrix is $M_A"
    I_A = UniformScaling(one(T)) - A_sym

    I_A⁻¹_sym = neumann_series(A_sym; Aⁿ_rewriter, maxn = something(max_An, size(A_sym, 1)))
    @assert M_I_A == UnitUpperTriangular && istriu(I_A⁻¹_sym) || M_I_A == UnitLowerTriangular && istril(I_A⁻¹_sym)
    if simplify
        verbose && @info "simplifying I - A⁻¹(θ)"
        # I_A⁻¹_sym = Symbolics.simplify.(I_A⁻¹_sym)
        Threads.@threads for i in eachindex(I_A⁻¹_sym.nzval)
            I_A⁻¹_sym.nzval[i] = Symbolics.simplify(I_A⁻¹_sym.nzval[i])
        end
    end
    verbose && @info "compiling in-place I_A⁻¹(θ) = I - A(θ)⁻¹"

    I_A⁻¹_eval, I_A⁻¹_eval! = Symbolics.build_function(I_A⁻¹_sym, pars, expression=Val{false})
    verbose && @info "compiling in-place I_Aoo(θ) = F × (I - A(θ)) × Fᵀ"
    I_Aoo_sym = I_A[observed_var_indices(ram), observed_var_indices(ram)]
    I_Aoo_eval, I_Aoo_eval! = Symbolics.build_function(I_Aoo_sym, pars, expression=Val{false})
    verbose && @info "compiling in-place I_All(θ) = L × (I - A(θ)) × Lᵀ"
    I_All_sym = I_A[latent_var_indices(ram), latent_var_indices(ram)]
    I_All_eval, I_All_eval! = Symbolics.build_function(I_All_sym, pars, expression=Val{false})
    verbose && @info "compiling in-place I_Aol(θ) = F × (I - A(θ)) × Lᵀ"
    I_Aol_sym = I_A[observed_var_indices(ram), latent_var_indices(ram)]
    I_Aol_eval, I_Aol_eval! = Symbolics.build_function(I_Aol_sym, pars, expression=Val{false})
    #let ctxt = IOContext(stdout, :compact => false)
    #    print("(I-Aol)="); show(ctxt, "text/plain", I_Aol_sym); println()
    #end

    randpars = randn(T, npars)
    # materialize sparse I_A submatrices
    I_Aoo_pre = M_I_A(I_Aoo_eval(randpars))
    @assert M_I_A == UnitLowerTriangular && istril(I_Aoo_pre) || M_I_A == UnitUpperTriangular && istriu(I_Aoo_pre)
    I_All_pre = M_I_A(I_All_eval(randpars))
    @assert M_I_A == UnitLowerTriangular && istril(I_All_pre) || M_I_A == UnitUpperTriangular && istriu(I_All_pre)
    I_Aol_pre = I_Aol_eval(randpars)
    I_A⁻¹_pre = M_I_A(I_A⁻¹_eval(randpars))
    F⨉I_A⁻¹_pre = F * I_A⁻¹_pre # not square, so cannot be triangular
    Σ_pre = Symmetric(zeros(T, nobs, nobs))
    S_par = materialize(ram.S, params(ram))
    S_pre = Symmetric(sparse_materialize(ram.S, randpars))
    Sol_par = S_par[observed_var_indices(ram), latent_var_indices(ram)]
    Sol_nz = findall(!iszero, Sol_par)
    if !isempty(Sol_nz)
        error("S contains $(length(Sol_nz)) covariations between observed and latent variables, ",
              "RAMLargeSparse SEM implied model not applicable")
    end
    Soo_parr = ParamsArray{T}(S_par[observed_var_indices(ram), observed_var_indices(ram)], params(ram))
    Sll_parr = ParamsArray{T}(S_par[latent_var_indices(ram), latent_var_indices(ram)], params(ram))
    Soo_pre = Symmetric(sparse_materialize(Soo_parr, randpars))
    Sll_pre = Symmetric(sparse_materialize(Sll_parr, randpars))
    Soo_chol = nothing
    Sll_chol = nothing
    SparseArrays.CHOLMOD.@cholmod_param final_ll = true begin
        Soo_chol = SparseArrays.CHOLMOD.analyze(SparseArrays.CHOLMOD.Sparse(Soo_pre))
        Sll_chol = SparseArrays.CHOLMOD.analyze(SparseArrays.CHOLMOD.Sparse(Sll_pre))
    end
    F⨉I_A⁻¹⨉S_pre = similar(parent(Σ_pre), size(F)...) # A×S produces dense pattern

    if gradient_required
        ∇A = sparse_gradient(ram.A)
        ∇S = sparse_gradient(ram.S)
    else
        ∇A = nothing
        ∇S = nothing
    end

    # μ
    if !isnothing(ram.M)
        MS = HasMeanStructure
        M_pre = materialize(ram.M, randpars)
        ∇M = gradient_required ? sparse_gradient(ram.M) : nothing
        μ_pre = zeros(T, nobs)
    else
        MS = NoMeanStructure
        M_pre = nothing
        ∇M = nothing
        μ_pre = nothing
    end

    return RAMLargeSparse{MS, typeof(ram), T, typeof(Soo_parr),
                          typeof(I_Aoo_eval!), typeof(I_All_eval!), typeof(I_Aol_eval!), typeof(I_A⁻¹_eval!),
                          typeof(F),
                          SparseMatrixCSC{T, Int64}, typeof(I_A⁻¹_pre),
                          typeof(parent(Soo_pre)), typeof(parent(Σ_pre)),
                          typeof(∇A),
                          typeof(M_pre), typeof(∇M),
                          typeof(Soo_chol), typeof(Sll_chol)}(
        ram, Soo_parr, Sll_parr,
        I_Aoo_eval!, I_All_eval!, I_Aol_eval!, I_A⁻¹_eval!,
        F, M_pre,
        I_Aoo_pre, I_All_pre, I_Aol_pre,
        I_A⁻¹_pre, F⨉I_A⁻¹_pre,
        S_pre, μ_pre,
        ∇A, ∇S, ∇M,
        Soo_shift, Soo_pre, Soo_chol,
        Sll_shift, Sll_pre, Sll_chol,
        allow_indef_S, nothing, nothing,
        F⨉I_A⁻¹⨉S_pre, Σ_pre, nothing, nothing, nothing,
        copy(F⨉I_A⁻¹⨉S_pre),
        zeros(T, nobs, nobs),
        zeros(T, nlat, nobs), zeros(T, nlat, nobs),
        zeros(T, nlat, nlat),
        nothing
    )
end

############################################################################################
### objective, gradient, hessian
############################################################################################

function update!(targets::EvaluationTargets, implied::RAMLargeSparse, par)
    reset_covars!(implied)

    materialize!(implied.S, implied.ram.S, par)

    implied.I_A⁻¹_eval!(parent(implied.I_A⁻¹), par)
    mul!(implied.F⨉I_A⁻¹, implied.F, implied.I_A⁻¹)

    if MeanStructure(implied) === HasMeanStructure
        materialize!(implied.M, implied.ram.M, par)
        mul!(implied.μ, implied.F⨉I_A⁻¹, implied.M)
    end

    # required only for Σ⁻¹ and logdet(Σ), but requires par, which is not passed to function update_Σ⁻¹!()
    materialize!(implied.Sll, implied.Sll_parr, par)
    materialize!(implied.Soo, implied.Soo_parr, par)
    implied.I_Aoo_eval!(parent(implied.I_Aoo), par)
    implied.I_All_eval!(parent(implied.I_All), par)
    implied.I_Aol_eval!(parent(implied.I_Aol), par)
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

# updates Cholesky decomposition of So and Sl
function update_S_chol!(implied::RAMLargeSparse)
    if isnothing(implied._isposdef_S)
        SparseArrays.CHOLMOD.cholesky!(implied._Sll_chol, implied.Sll; shift=implied.Sll_shift, check=false)
        SparseArrays.CHOLMOD.cholesky!(implied._Soo_chol, implied.Soo; shift=implied.Soo_shift, check=false)
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
        implied._isposdef_S = _isposdef(implied._Sll_chol) && _isposdef(implied._Soo_chol) # Choleski succeeded
    end
end

# direct way of calculating Σoo cholesky from Σoo when S is not positive definite
function update_Σ_chol!(implied::RAMLargeSparse)
    if isnothing(implied._Σ_chol)
        implied._Σ_chol = cholesky!(copy!(implied._Σ⁻¹oo_buf, implied.Σ), check=false)
    end
end

function update_Σ!(implied::RAMLargeSparse)
    if isnothing(implied._Σ)
        implied._Σ = Symmetric(X_A_Xt!(implied._Σ_buf, implied.S, implied.F⨉I_A⁻¹,
                                       X_A_buf = implied.F⨉I_A⁻¹⨉S))
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
    if isposdef_S(implied) # "sparse" (faster?) path
        update_Σ⁻¹_sparse!(implied)
    elseif isposdef_Σ(implied) && implied.allow_indef_S # "dense" (slow) path
        update_Σ_chol!(implied)
        implied._logdet_Σ = logdet(implied._Σ_chol)
        implied._Σ⁻¹ = Symmetric(LinearAlgebra.inv!(implied._Σ_chol))
        implied._Σ_chol = nothing # invalidate Σ_chol since it got inverted
    else
        implied._logdet_Σ = NaN
        throw(LinearAlgebra.PosDefException(0))
    end
end

function update_Σ⁻¹_sparse!(implied::RAMLargeSparse)
    isposdef_S(implied) || throw(LinearAlgebra.PosDefException(0))
    # parent() because CHOLMOD dispatch does not support triangular matrices:
    # the dispatch goes to generic triangular matrices
    Soo⁻½⨉I_Aoo = implied._Soo_chol.PtL \ parent(implied.I_Aoo)
    Soo⁻½⨉I_Aol = implied._Soo_chol.PtL \ implied.I_Aol
    Sll⁻½⨉I_All = implied._Sll_chol.PtL \ parent(implied.I_All)

    # Σ⁻¹ = (I-A)ᵀ×(S⁻½×S⁻½)×(I-A) -- inverse of covariations of all variables
    # Σ⁻¹ = [Σ⁻¹oo Σ⁻¹lo'
    #        Σ⁻¹lo Σ⁻¹ll]
    Σ⁻¹lo = mul!(implied._Σ⁻¹lo_buf, Soo⁻½⨉I_Aol', Soo⁻½⨉I_Aoo)
    Σ⁻¹ll = Symmetric(Xt_X!(implied._Σ⁻¹ll_buf, Soo⁻½⨉I_Aol))
    Xt_X!(parent(Σ⁻¹ll), Sll⁻½⨉I_All, 1, 1)
    #let ctxt = IOContext(stdout, :compact => false)
    #     print("Soo="); show(ctxt, "text/plain", implied.Soo); println()
    #     print("Sll="); show(ctxt, "text/plain", implied.Sll); println()
    #     print("I_Aol="); show(ctxt, "text/plain", implied.I_Aol); println()
    #     print("Soo⁻½⨉I_Aol="); show(ctxt, "text/plain", Soo⁻½⨉I_Aol); println()
    #     print("I_Aol="); show(ctxt, "text/plain", implied.I_Aol); println()
    #     print("I_All="); show(ctxt, "text/plain", implied.I_All); println()
    #    print("Σ⁻¹ll="); show(ctxt, "text/plain", Σ⁻¹ll); println()
    #end
    if implied.Sll_shift != 0.0
        # add shift to the diagonal of Σ⁻¹ll
        @inbounds for i in diagind(Σ⁻¹ll)
            parent(Σ⁻¹ll)[i] += implied.Sll_shift
        end
    end
    Σ⁻¹ll_chol = cholesky!(Σ⁻¹ll)
    @assert isposdef(Σ⁻¹ll_chol)
    # logdet(Σoo) = logdet(Σ) + logdet(Σ⁻¹ll), since Σ⁻¹ll is complement to Σoo
    # logdet(Σ) = logdet(So) + logdet(Sl) + "logdet"(I_A⁻¹o)
    # "logdet"(I_A⁻¹o) is always zero since it is "unit triangular" (not square)
    implied._logdet_Σ = logdet(implied._Soo_chol) + logdet(implied._Sll_chol) + logdet(Σ⁻¹ll_chol)
    Σ⁻¹ll⁻½⨉Σ⁻¹lo = ldiv!(Σ⁻¹ll_chol.L, Σ⁻¹lo)
    # use Schur complement formula to calculate Σoo⁻¹
    Σ⁻¹oo = Xt_X!(parent(implied._Σ⁻¹oo_buf), Soo⁻½⨉I_Aoo)
    Σoo⁻¹ = Xt_X!(Σ⁻¹oo, Σ⁻¹ll⁻½⨉Σ⁻¹lo, -1, 1)
    implied._Σ⁻¹ = Symmetric(Σoo⁻¹)
    # let ctxt = IOContext(stdout, :compact => false)
    #     print("Σoo⁻¹="); show(ctxt, "text/plain", Σoo⁻¹); println()
    # end
end

Base.propertynames(implied::RAMLargeSparse, private::Bool=false) =
    (:logdet_Σ, :Σ, :Σ⁻¹,
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
    elseif name == :ram_matrices
        return implied.ram
    else
        return getfield(implied, name)
    end
end
