############################################################################################
### Types
############################################################################################
@doc raw"""
Model implied covariance and means via RAM notation.

# Constructor

    RAM(;
        specification,
        meanstructure = false,
        gradient = true,
        kwargs...)

# Arguments
- `specification`: either a `RAMMatrices` or `ParameterTable` object
- `meanstructure::Bool`: does the model have a meanstructure?
- `gradient::Bool`: is gradient-based optimization used

# Extended help

## Implementation
Subtype of `SemImply`.

## RAM notation

The model implied covariance matrix is computed as
```math
    \Sigma = F(I-A)^{-1}S(I-A)^{-T}F^T
```
and for models with a meanstructure, the model implied means are computed as
```math
    \mu = F(I-A)^{-1}M
```

## Interfaces
- `params(::RAM) `-> Dict containing the parameter labels and their position
- `nparams(::RAM)` -> Number of parameters

- `Σ(::RAM)` -> model implied covariance matrix
- `μ(::RAM)` -> model implied mean vector

RAM matrices for the current parameter values:
- `A(::RAM)`
- `S(::RAM)`
- `F(::RAM)`
- `M(::RAM)`

Jacobians of RAM matrices w.r.t to the parameter vector `θ`
- `∇A(::RAM)` -> ``∂vec(A)/∂θᵀ``
- `∇S(::RAM)` -> ``∂vec(S)/∂θᵀ``
- `∇M(::RAM)` = ``∂M/∂θᵀ``

Vector of indices of each parameter in the respective RAM matrix:
- `A_indices(::RAM)`
- `S_indices(::RAM)`
- `M_indices(::RAM)`

Additional interfaces
- `F⨉I_A⁻¹(::RAM)` -> ``F(I-A)^{-1}``
- `F⨉I_A⁻¹S(::RAM)` -> ``F(I-A)^{-1}S``
- `I_A(::RAM)` -> ``I-A``
- `has_meanstructure(::RAM)` -> `Val{Bool}` does the model have a meanstructure?

Only available in gradient! calls:
- `I_A⁻¹(::RAM)` -> ``(I-A)^{-1}``
"""
mutable struct RAM{MS, SPEC, T, M_A, M_FA, M_S, M_F, M_SGM, M_ISGM, M_CHOL, CHOL, V_M, GM, GM_M} <: SemImply{MS, ExactHessian}
    ram_matrices::SPEC

    A::M_A
    S::M_S
    F::M_F
    μ::V_M
    M::V_M

    F⨉I_A⁻¹::M_FA
    F⨉I_A⁻¹S::M_FA

    I_A::M_A
    I_A⁻¹::M_A

    Σ::M_SGM

    _Σ_chol_buf::M_CHOL
    _Σ_chol::Union{CHOL, Nothing}
    _isposdef_Σ::Union{Bool, Nothing}
    _logdet_Σ::Union{T, Nothing}
    _Σ⁻¹::Union{M_ISGM, Nothing}

    ∇A::GM
    ∇S::GM
    ∇M::GM_M
end

############################################################################################
### Constructors
############################################################################################

function RAM(spec::SemSpecification;
    #vech = false,
    gradient_required = true,
    sparse_S::Bool = true
)
    ram_matrices = convert(RAMMatrices, spec)

    # get dimensions of the model
    n_par = nparams(ram_matrices)
    n_obs = nobserved_vars(ram_matrices)
    n_var = nvars(ram_matrices)

    #preallocate arrays
    T = Float64
    rand_params = randn(T, n_par)
    A_pre = check_acyclic(materialize(ram_matrices.A, rand_params))
    S_pre = Symmetric((sparse_S ? sparse_materialize : materialize)(ram_matrices.S, rand_params))
    F = copy(ram_matrices.F)

    # pre-allocate some matrices
    I_A = typeof(A_pre)(I - parent(A_pre))
    F⨉I_A⁻¹ = F * A_pre
    F⨉I_A⁻¹S = similar(F⨉I_A⁻¹)
    Σ = Symmetric(zeros(T, n_obs, n_obs))

    if gradient_required
        ∇A = sparse_gradient(ram_matrices.A)
        ∇S = sparse_gradient(ram_matrices.S)
    else
        ∇A = nothing
        ∇S = nothing
    end

    # μ
    if !isnothing(ram_matrices.M)
        MS = HasMeanStructure
        M_pre = materialize(ram_matrices.M, rand_params)
        ∇M = gradient_required ? sparse_gradient(ram_matrices.M) : nothing
        μ = zeros(n_obs)
    else
        MS = NoMeanStructure
        M_pre = nothing
        μ = nothing
        ∇M = nothing
    end

    Σ_chol = cholesky!(convert(Matrix{T}, I(size(Σ, 1))))

    return RAM{MS, typeof(ram_matrices), T,
               typeof(A_pre), typeof(F⨉I_A⁻¹), typeof(S_pre), typeof(F),
               typeof(Σ), typeof(Symmetric(Σ_chol.factors)), typeof(Σ_chol.factors), typeof(Σ_chol),
               typeof(M_pre), typeof(∇A), typeof(∇M)}(
        ram_matrices,
        A_pre,
        S_pre,
        F,
        μ, M_pre,

        F⨉I_A⁻¹, F⨉I_A⁻¹S,
        I_A, similar(I_A),

        Σ, Σ_chol.factors, nothing, nothing, nothing, nothing,

        ∇A,
        ∇S,
        ∇M
    )
end

############################################################################################
### methods
############################################################################################

function update!(targets::EvaluationTargets, imply::RAM, params)
    reset_covars!(imply)

    materialize!(imply.A, imply.ram_matrices.A, params)
    materialize!(imply.S, imply.ram_matrices.S, params)
    if !isnothing(imply.M)
        materialize!(imply.M, imply.ram_matrices.M, params)
    end

    @inbounds for (j, I_Aj, Aj) in zip(axes(imply.A, 2), eachcol(parent(imply.I_A)), eachcol(imply.A))
        for i in axes(imply.A, 1)
            I_Aj[i] = ifelse(i == j, 1, 0) - Aj[i]
        end
    end

    if is_gradient_required(targets) || is_hessian_required(targets)
        try
            imply.I_A⁻¹ = LinearAlgebra.inv!(factorize(imply.I_A))
        catch e
            if e isa SingularException
                @warn "Singular I-A matrix in RAM model, trying pinv()"
                # if fails (e.g. because of bad initial values), try pseudoinverse
                imply.I_A⁻¹ = LinearAlgebra.pinv(imply.I_A)
            else
                rethrow(e)
            end
        end
        mul!(imply.F⨉I_A⁻¹, imply.F, imply.I_A⁻¹)
    else
        copyto!(imply.F⨉I_A⁻¹, imply.F)
        rdiv!(imply.F⨉I_A⁻¹, factorize(imply.I_A))
    end

    X_A_Xt!(parent(imply.Σ), imply.S, imply.F⨉I_A⁻¹, X_A_buf = imply.F⨉I_A⁻¹S)

    if MeanStructure(imply) === HasMeanStructure
        mul!(imply.μ, imply.F⨉I_A⁻¹, imply.M)
    end

end

############################################################################################
### Recommended methods
############################################################################################

function update_observed(imply::RAM, observed::SemObserved; kwargs...)
    if n_man(observed) == size(imply.Σ, 1)
        return imply
    else
        return RAM(;observed = observed, kwargs...)
    end
end
