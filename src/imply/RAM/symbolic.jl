############################################################################################
### Types
############################################################################################
@doc raw"""
Subtype of `SemImply` that implements the RAM notation with symbolic precomputation.

# Constructor

    RAMSymbolic(;specification,
        vech = false,
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
- `params(::RAMSymbolic) `-> vector of parameter names
- `nparams(::RAMSymbolic)` -> number of parameters

- `Σ(::RAMSymbolic)` -> model implied covariance matrix
- `μ(::RAMSymbolic)` -> model implied mean vector

Jacobians (only available in gradient! calls)
- `∇Σ(::RAMSymbolic)` -> ``∂vec(Σ)/∂θᵀ``
- `∇μ(::RAMSymbolic)` -> ``∂μ/∂θᵀ``

- `∇Σ_function(::RAMSymbolic)` -> function to overwrite `∇Σ` in place,
    i.e. `∇Σ_function(∇Σ, θ)`. Normally, you do not want to use this but simply
    query `∇Σ(::RAMSymbolic)`.

Hessians
The computation of hessians is more involved, and uses the "chain rule for
hessian matrices".
Therefore, we desribe it at length in the mathematical appendix of the online documentation,
and the relevant interfaces are omitted here.

Additional interfaces
- `has_meanstructure(::RAMSymbolic)` -> `Val{Bool}` does the model have a meanstructure?

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
mutable struct RAMSymbolic{MS, SPEC, T, F1, F2, F3, F4, F5, M_SGM, M_ISGM, M_CHOL, CHOL, GM, V, GM2} <: SemImplySymbolic{MS,ExactHessian}
    ram_matrices::SPEC
    Σ_eval!::F1
    ∇Σ_eval!::F2
    ∇²Σ_eval!::F3

    μ_eval!::F4
    ∇μ_eval!::F5

    Σ::M_SGM

    _Σ_chol_buf::M_CHOL
    _Σ_chol::Union{CHOL, Nothing}
    _isposdef_Σ::Union{Bool, Nothing}
    _logdet_Σ::Union{T, Nothing}
    _Σ⁻¹::Union{M_ISGM, Nothing}

    ∇Σ::GM
    μ::V
    ∇μ::GM2
end

############################################################################################
### Constructors
############################################################################################

RAMSymbolic{MS}(args...) where MS <: MeanStructure = RAMSymbolic{MS, map(typeof, args)...}(args...)

function RAMSymbolic(spec::SemSpecification;
    vech::Bool = false,
    gradient::Bool = true,
    hessian::Bool = false,
    approximate_hessian::Bool = false,
    simplify::Bool = true,
    Aⁿ_rewriter = nothing,
    max_An::Union{Nothing, Integer} = nothing,
    verbose::Bool = false
)
    ram_matrices = convert(RAMMatrices, spec)

    n_par = nparams(ram_matrices)
    par = (Symbolics.@variables θ[1:n_par])[1]
    parvec = [par...]

    A = sparse_materialize(Num, ram_matrices.A, par)
    S = sparse_materialize(Num, ram_matrices.S, par)
    F = ram_matrices.F

    verbose && @info "computing symbolic I - A⁻¹"
    I_A⁻¹ = neumann_series(A; Aⁿ_rewriter, maxn = something(max_An, size(A, 1)))
    F⨉I_A⁻¹ = F*I_A⁻¹

    # Σ
    verbose && @info "computing symbolic Σ"
    Σ_sym = convert(Matrix, F⨉I_A⁻¹*S*permutedims(F⨉I_A⁻¹))
    vech && (Σ_sym = SEM.vech(Σ_sym))
    if simplify
        verbose && @info "simplyfing Σ"
        # Σ_sym = Symbolics.simplify.(Σ_sym)
        Threads.@threads for i in eachindex(Σ_sym)
            Σ_sym[i] = Symbolics.simplify(Σ_sym[i])
        end
    end

    verbose && @info "compiling Σ(θ)"
    Σ_eval! = Symbolics.build_function(Σ_sym, par, expression=Val{false})[2]
    T = Float64
    Σ = zeros(T, size(Σ_sym))
    vech || (Σ = Symmetric(Σ))
    precompile(Σ_eval!, (typeof(parent(Σ)), Vector{T}))

    # ∇Σ
    if gradient
        verbose && @info "computing symbolic ∇Σ"
        ∇Σ_sym = Symbolics.sparsejacobian(vec(Array(Σ_sym)), parvec; simplify)
        verbose && @info "compiling ∇Σ(θ)"
        ∇Σ_eval! = Symbolics.build_function(∇Σ_sym, par, expression=Val{false})[2]
        constr = findnz(∇Σ_sym)
        ∇Σ = sparse(constr[1], constr[2], fill(1.0, nnz(∇Σ_sym)), size(∇Σ_sym)...)
        precompile(∇Σ_eval!, (typeof(∇Σ), Vector{Float64}))
    else
        ∇Σ_eval! = nothing
        ∇Σ = nothing
    end

    if hessian && !approximate_hessian
        verbose && @info "computing symbolic ∇²Σ"
        @variables J[1:length(Σ_sym)]
        ∇²Σ_sym = zeros(Num, n_par, n_par)
        for (J_i, Σ_sym_i) in zip(J, vec(Σ_sym))
            ∇²Σ_sym_i = Symbolics.sparsehessian(Σ_sym_i, parvec; simplify)
            ∇²Σ_sym .+= J_i * ∇²Σ_sym_i
        end

        verbose && @info "compiling ∇²Σ(J, θ)"
        ∇²Σ_eval! = Symbolics.build_function(∇²Σ_sym, J, par, expression=Val{false})[2]
    else
        ∇²Σ_eval! = nothing
    end

    # μ
    if !isnothing(ram_matrices.M)
        MS = HasMeanStructure

        M = materialize(Num, ram_matrices.M, par)
        verbose && @info "computing symbolic μ"
        μ_sym = convert(Vector{Num}, F⨉I_A⁻¹*M)
        if simplify
            verbose && @info "simplyfing μ"
            # μ_sym = Symbolics.simplify.(μ_sym)
            Threads.@threads for i in eachindex(μ_sym)
                μ_sym[i] = Symbolics.simplify(μ_sym[i])
            end
        end

        verbose && @info "compiling μ(θ)"
        μ_eval! = Symbolics.build_function(μ_sym, par, expression=Val{false})[2]
        μ = zeros(T, size(μ_sym))
        if gradient
            verbose && @info "computing symbolic ∇μ"
            ∇μ_sym = Symbolics.jacobian(μ_sym, parvec)
            verbose && @info "compiling ∇μ(θ)"
            ∇μ_eval! = Symbolics.build_function(∇μ_sym, par, expression=Val{false})[2]
            ∇μ = zeros(T, length(μ), n_par)
        else
            ∇μ_eval! = nothing
            ∇μ = nothing
        end
    else
        MS = NoMeanStructure
        μ_eval! = nothing
        μ = nothing
        ∇μ_eval! = nothing
        ∇μ = nothing
    end

    Σ_chol = cholesky!(convert(Matrix{T}, I(size(Σ, 1))))

    return RAMSymbolic{MS, typeof(ram_matrices), T,
        typeof(Σ_eval!), typeof(∇Σ_eval!), typeof(∇²Σ_eval!),
        typeof(μ_eval!), typeof(∇μ_eval!),
        typeof(Σ), typeof(Symmetric(Σ_chol.factors)), typeof(Σ_chol.factors), typeof(Σ_chol),
        typeof(∇Σ), typeof(μ), typeof(∇μ)
    }(
        ram_matrices,
        Σ_eval!, ∇Σ_eval!, ∇²Σ_eval!,
        μ_eval!, ∇μ_eval!,
        Σ, Σ_chol.factors, nothing, nothing, nothing, nothing,
        ∇Σ,
        μ,
        ∇μ,
    )
end

############################################################################################
### objective, gradient, hessian
############################################################################################

function update!(targets::EvaluationTargets, imply::RAMSymbolic, par)
    reset_covars!(imply)

    imply.Σ_eval!(parent(imply.Σ), par)
    if MeanStructure(imply) === HasMeanStructure
        imply.μ_eval!(imply.μ, par)
    end

    if is_gradient_required(targets) || is_hessian_required(targets)
        imply.∇Σ_eval!(imply.∇Σ, par)
        if MeanStructure(imply) === HasMeanStructure
            imply.∇μ_eval!(imply.∇μ, par)
        end
    end
    # note: ∇²Σ is evaluated within the evaluate!() of a specific loss term
    # (using ∇²Σ_eval!(J, par)), since the Jacobian (J) depends on the input data (Σₒ)
end

############################################################################################
### Recommended methods
############################################################################################

function update_observed(imply::RAMSymbolic, observed::SemObserved; kwargs...)
    if n_man(observed) == size(imply.Σ, 1)
        return imply
    else
        return RAMSymbolic(;observed = observed, kwargs...)
    end
end
