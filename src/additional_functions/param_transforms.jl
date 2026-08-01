"""
    CovarianceTransforms(covariance_indices, variance_sources, nparams)
    CovarianceTransforms(S::ParamsMatrix; skip_params=nothing)

Index-based relationships used to convert correlation-like optimizer parameters to
covariances. Entry `k` of `variance_sources` contains the two variance parameter
indices followed by their fixed values for `covariance_indices[k]`. A zero variance
index denotes that the corresponding fixed value is used.

The `S`-matrix constructor infers these relationships from its structure. A parameter
that occurs on the diagonal is treated as a variance parameter, even if it is reused
off the diagonal; every parameter that occurs only off the diagonal is treated as a
covariance parameter. Parameters listed by index in `skip_params` are excluded from
covariance-transform inference, for example because they already have suitable scalar
transformations. Reusing any remaining covariance parameter for different pairs of
diagonal variances is an error.
"""
struct CovarianceTransforms{T <: Real}
    covariance_indices::Vector{Int}
    variance_sources::Vector{Tuple{Int, Int, T, T}}

    function CovarianceTransforms{T}(
        covariance_indices::Vector{Int},
        variance_sources::Vector{Tuple{Int, Int, T, T}},
    ) where {T <: Real}
        return new{T}(covariance_indices, variance_sources)
    end
end

function CovarianceTransforms(
    covariance_indices::AbstractVector{<:Integer},
    variance_sources::AbstractVector{<:Tuple{Integer, Integer, Real, Real}},
    nparams::Integer,
)
    check_covariance_transforms(covariance_indices, variance_sources, nparams)

    fixed_types = Type[
        typeof(source[position])
        for source in variance_sources for position in 3:4
    ]
    T = isempty(fixed_types) ? Float64 : float(promote_type(fixed_types...))
    return CovarianceTransforms{T}(
        convert(Vector{Int}, covariance_indices),
        convert(Vector{Tuple{Int, Int, T, T}}, variance_sources),
    )
end

function check_covariance_transforms(
    covariance_indices::AbstractVector{<:Integer},
    variance_sources::AbstractVector{<:Tuple{Integer, Integer, Real, Real}},
    nparams::Integer,
)
    nparams >= 0 || throw(ArgumentError(
        "The number of parameters must be nonnegative, got $nparams"))
    length(variance_sources) == length(covariance_indices) ||
        throw(DimensionMismatch("variance_sources must have length " *
                                "$(length(covariance_indices))"))

    for (cov_ix, (var1_ix, var2_ix, fixed_var1, fixed_var2)) in
        zip(covariance_indices, variance_sources)
        1 <= cov_ix <= nparams || throw(ArgumentError(
            "Covariance parameter index $cov_ix must be between 1 and $nparams"))
        if var1_ix < 0
            throw(ArgumentError(
                "Covariance parameter index $cov_ix has a negative variance index 1: " *
                "$var1_ix"))
        elseif var1_ix > 0
            var1_ix > nparams && throw(ArgumentError(
                "Covariance parameter index $cov_ix has a variance index 1 " *
                "($var1_ix) that exceeds the number of parameters ($nparams)"))
            var1_ix == cov_ix && throw(ArgumentError(
                "Covariance parameter index $cov_ix cannot also be its own " *
                "variance index 1"))
        else
            isfinite(fixed_var1) && fixed_var1 > 0 ||
            throw(ArgumentError(
                "Covariance parameter index $cov_ix has a nonpositive or non-finite " *
                "fixed variance for variance index 1: $fixed_var1"))
        end
        if var2_ix < 0
            throw(ArgumentError(
                "Covariance parameter index $cov_ix has a negative variance index 2: " *
                "$var2_ix"))
        elseif var2_ix > 0
            var2_ix > nparams && throw(ArgumentError(
                "Covariance parameter index $cov_ix has a variance index 2 " *
                "($var2_ix) that exceeds the number of parameters ($nparams)"))
            var2_ix == cov_ix && throw(ArgumentError(
                "Covariance parameter index $cov_ix cannot also be its own " *
                "variance index 2"))
        else
            isfinite(fixed_var2) && fixed_var2 > 0 ||
            throw(ArgumentError(
                "Covariance parameter index $cov_ix has a nonpositive or non-finite " *
                "fixed variance for variance index 2: $fixed_var2"))
        end
    end
    allunique(covariance_indices) || throw(ArgumentError(
        "Covariance parameter indices must be unique, got $covariance_indices"))
    return nothing
end

CovarianceTransforms(::Nothing = nothing) =
    CovarianceTransforms(
        Int[], Tuple{Int, Int, Float64, Float64}[], 0)

function CovarianceTransforms(
    S::ParamsMatrix;
    skip_params::Union{AbstractVector{<:Integer}, Nothing} = nothing,
)
    size(S, 1) == size(S, 2) ||
        throw(DimensionMismatch("The RAM S matrix must be square"))

    skip_param_mask = falses(nparams(S))
    if !isnothing(skip_params)
        for par_ix in skip_params
            1 <= par_ix <= nparams(S) || throw(ArgumentError(
                "Skipped parameter index $par_ix must be between 1 and $(nparams(S))"))
            skip_param_mask[par_ix] = true
        end
    end

    nvars = size(S, 1)
    S_ixs = CartesianIndices(size(S))
    diag_par_ixs = zeros(Int, nvars)
    fixed_diag_vals = zeros(eltype(S), nvars)
    diag_param_mask = falses(nparams(S))
    for param_ix in 1:nparams(S)
        for S_ix in param_occurences(S, param_ix)
            row, column = Tuple(S_ixs[S_ix])
            if row == column
                diag_par_ixs[row] = param_ix
                diag_param_mask[param_ix] = true
            end
        end
    end
    for (linear_ix, _, value) in S.constants
        row, col = Tuple(S_ixs[linear_ix])
        row == col && (fixed_diag_vals[row] = value)
    end

    cov_ixs = Int[]
    variance_sources = Tuple{Int, Int, eltype(S), eltype(S)}[]
    for par_ix in 1:nparams(S)
        (diag_param_mask[par_ix] || skip_param_mask[par_ix]) && continue
        var_srcs = nothing
        for linear_ix in param_occurences(S, par_ix)
            row, col = Tuple(S_ixs[linear_ix])
            row == col && continue
            diag_par_row, diag_par_col = diag_par_ixs[row], diag_par_ixs[col]
            var_src_row = (iszero(diag_par_row), diag_par_row, fixed_diag_vals[row])
            var_src_col = (iszero(diag_par_col), diag_par_col, fixed_diag_vals[col])
            cur_var_srcs = isless(var_src_col, var_src_row) ?
                (var_src_col, var_src_row) : (var_src_row, var_src_col)
            if isnothing(var_srcs)
                var_srcs = cur_var_srcs
            else
                var_srcs == cur_var_srcs || throw(ArgumentError(
                    "Covariance parameter index $par_ix refers to multiple " *
                    "pairs of diagonal variances"))
            end
        end
        isnothing(var_srcs) && continue # not a covariance param
        push!(cov_ixs, par_ix)
        push!(variance_sources, (
            var_srcs[1][2], var_srcs[2][2],
            var_srcs[1][3], var_srcs[2][3],
        ))
    end
    return CovarianceTransforms(cov_ixs, variance_sources, nparams(S))
end

CovarianceTransforms(
    S::AbstractMatrix,
    params::AbstractVector{Symbol},
    skip_params::Union{AbstractVector{<:Integer}, Nothing} = nothing,
) = CovarianceTransforms(ParamsMatrix{Float64}(S, params); skip_params)

Base.isempty(transforms::CovarianceTransforms) = isempty(transforms.covariance_indices)

function _scale_covariances!(model_vals, transforms::CovarianceTransforms)
    (; covariance_indices, variance_sources) = transforms
    @inbounds for k in eachindex(covariance_indices)
        _, _, covscale = var1_var2_covscale(model_vals, variance_sources[k])
        model_vals[covariance_indices[k]] *= covscale
    end
    return model_vals
end

function _unscale_covariances!(
    unconstrained_vals, transforms::CovarianceTransforms,
    model_vals
)
    (; covariance_indices, variance_sources) = transforms
    @inbounds for k in eachindex(covariance_indices)
        _, _, covscale = var1_var2_covscale(model_vals, variance_sources[k])
        covariance_index = covariance_indices[k]
        correlation = model_vals[covariance_index] / covscale
        (-1 < correlation < 1) ||
            throw(DomainError(model_vals[covariance_index],
                "Starting covariance at parameter index $covariance_index must imply " *
                "a correlation strictly between -1 and 1"))
        unconstrained_vals[covariance_index] = correlation
    end
    return unconstrained_vals
end

"""
    ParamTransforms(params, transforms[, covariance_transforms])

Transformations between two parameter spaces:

- **model values** are parameters on the scale used by the RAM matrices and losses;
- **unconstrained values** are coordinates in `ℝⁿ` passed to the optimizer.

`transform_params` maps unconstrained values to model values, and
`inverse_transform_params` maps model values to unconstrained values. `transforms`
can be a vector in `params` order or a dictionary keyed by parameter name.
Missing dictionary entries use `TransformVariables.asℝ`.

Scalar transformations are applied first. For indices listed in
`covariance_transforms`, the scalar result is interpreted as a correlation `ρ` and
replaced by `ρ * sqrt(variance1 * variance2)`.
"""
struct ParamTransforms{G <: Tuple, C <: CovarianceTransforms}
    transforms::Vector{Any}
    groups::G
    covariance_transforms::C
end

struct ParamTransformGroup{T}
    indices::Vector{Int}
    transforms::Vector{T}
    flips::Vector{Bool}
end

function is_flipped(transform)
    TransformVariables.dimension(transform) == 1 ||
        throw(ArgumentError("Parameter transformations must have dimension 1, got " *
                            "$(TransformVariables.dimension(transform)) for $(typeof(transform))"))
    lower = TransformVariables.transform(transform, -1.0)
    upper = TransformVariables.transform(transform, 1.0)
    (lower isa Number && upper isa Number) ||
        throw(ArgumentError("Parameter transformations must map a scalar to a scalar, got " *
                            "$(typeof(lower)) from $(typeof(transform))"))
    if upper > lower
        return false
    elseif upper < lower
        return true
    end
    throw(ArgumentError("Could not determine the direction of parameter transformation " *
                        "$(typeof(transform))"))
end

function ParamTransforms(
    params::AbstractVector{Symbol}, scalar_transforms,
    covariance_transforms::CovarianceTransforms = CovarianceTransforms(),
)
    check_param_transforms(params, scalar_transforms, covariance_transforms)
    trfs = if scalar_transforms isa AbstractDict
        Any[get(scalar_transforms, param, TransformVariables.asℝ) for param in params]
    else
        convert(Vector{Any}, scalar_transforms)
    end
    trf_type_map = Dict{Type, Vector{Int}}()
    for (i, trf) in enumerate(trfs)
        trf_group_ixs = get!(() -> Int[], trf_type_map, typeof(trf))
        push!(trf_group_ixs, i)
    end
    trf_groups = Tuple(ParamTransformGroup(
        trf_ixs,
        trf_type[trfs[i] for i in trf_ixs],
        [is_flipped(trfs[i]) for i in trf_ixs]
    ) for (trf_type, trf_ixs) in pairs(trf_type_map))
    return ParamTransforms(trfs, trf_groups, covariance_transforms)
end

function check_param_transforms(
    params::AbstractVector{Symbol}, scalar_transforms,
    covariance_transforms::CovarianceTransforms = CovarianceTransforms(),
)
    if scalar_transforms isa AbstractDict
        unknown = setdiff(keys(scalar_transforms), params)
        isempty(unknown) || throw(ArgumentError(
            "Transformations supplied for unknown parameters: $(join(unknown, ", "))"))
    elseif scalar_transforms isa AbstractVector
        length(scalar_transforms) == length(params) ||
            throw(DimensionMismatch("The number of parameter transformations " *
                                    "($(length(scalar_transforms))) does not match the number of model " *
                                    "parameters ($(length(params)))"))
    else
        throw(ArgumentError(
            "Parameter transformations must be an AbstractVector or AbstractDict, " *
            "got $(typeof(scalar_transforms))"))
    end
    return nothing
end

"""
    allidentity(transforms)

Return whether all parameter transformations are identity transformations.
Covariance scaling is non-identity regardless of its scalar transformation.
"""
allidentity(transforms::ParamTransforms) =
    isempty(transforms.covariance_transforms) &&
    all(==(TransformVariables.asℝ), transforms.transforms)

"""
    nonidentity_transformed_params(params, transforms)

Return the set of parameter names whose transformations affect model evaluation.
Covariance parameters are included even when their scalar transform is identity,
because their covariance scaling is still non-identity.
"""
function nonidentity_transformed_params(
    params::AbstractVector{Symbol}, transforms::ParamTransforms)
    coupled = Set(transforms.covariance_transforms.covariance_indices)
    return Set{Symbol}(
        param
        for (i, (param, transform)) in enumerate(
            zip(params, transforms.transforms))
        if transform != TransformVariables.asℝ || i in coupled
    )
end

function check_params_vector(destination, transforms::ParamTransforms, source)
    length(source) == length(transforms.transforms) ||
        throw(DimensionMismatch("The parameter vector length ($(length(source))) does not match " *
                                "the number of transformations ($(length(transforms.transforms)))"))
    length(destination) == length(source) ||
        throw(DimensionMismatch("Source and destination parameter vectors have different lengths"))
    return nothing
end

"""
    merge_param_transforms(target_params, transform_specs)

Merge parameter transformations into `target_params` order. Each entry of
`transform_specs` is `source_params => transforms`, where `transforms` may be
`nothing`. Parameters absent from a source do not participate in consistency
checking; parameters shared by multiple sources must have identical scalar and
covariance transformations.
"""
function merge_param_transforms(
    target_params::AbstractVector{Symbol}, transform_specs
)
    allunique(target_params) ||
        throw(ArgumentError("Target parameter names must be unique"))
    target_par2ix = Dict(param => i for (i, param) in enumerate(target_params))
    scalar_transforms = Any[TransformVariables.asℝ for _ in target_params]
    scalar_seen = falses(length(target_params))
    merged_cov_trfs =
        Dict{Int, Union{Nothing, Tuple{Int, Int, Real, Real}}}()

    for (src_pars, trfs) in transform_specs
        src_pars isa AbstractVector{Symbol} || throw(ArgumentError(
            "Transformation source parameters must be an AbstractVector{Symbol}"))
        allunique(src_pars) ||
            throw(ArgumentError("Transformation source parameter names must be unique"))
        if !isnothing(trfs) &&
                length(trfs.transforms) != length(src_pars)
            throw(DimensionMismatch(
                "The number of parameter transformations " *
                "($(length(trfs.transforms))) does not match the number " *
                "of source parameters ($(length(src_pars)))"))
        end

        converted_src_cov_trfs = Dict{Int, Tuple{Int, Int, Real, Real}}()
        if !isnothing(trfs)
            cov_trf = trfs.covariance_transforms
            for (cov_ix, var_srcs) in zip(cov_trf.covariance_indices, cov_trf.variance_sources)
                conv_srcs = map(((var_srcs[1], var_srcs[3]),
                                 (var_srcs[2], var_srcs[4]))) do (src_var_ix, src_val)
                    iszero(src_var_ix) && return (true, 0, src_val)
                    var_par = src_pars[src_var_ix]
                    target_var_ix = get(target_par2ix, var_par, 0)
                    iszero(target_var_ix) && throw(ArgumentError(
                        "Variance parameter :$var_par is absent from the target parameters"))
                    return (false, target_var_ix, zero(src_val))
                end
                if isless(conv_srcs[2], conv_srcs[1])
                    conv_srcs = (conv_srcs[2], conv_srcs[1])
                end
                converted_src_cov_trfs[cov_ix] = (conv_srcs[1][2], conv_srcs[2][2], conv_srcs[1][3], conv_srcs[2][3])
            end
        end

        for (src_ix, param) in enumerate(src_pars)
            target_ix = get(target_par2ix, param, 0)
            iszero(target_ix) && throw(ArgumentError(
                "Parameter :$param is absent from the target parameters"))
            trf = isnothing(trfs) ? TransformVariables.asℝ : trfs.transforms[src_ix]
            if scalar_seen[target_ix] && !isequal(scalar_transforms[target_ix], trf)
                throw(ArgumentError("Conflicting scalar transforms for parameter :$param"))
            end
            scalar_transforms[target_ix] = trf
            scalar_seen[target_ix] = true

            cov_var_srcs = get(converted_src_cov_trfs, src_ix, nothing)
            if haskey(merged_cov_trfs, target_ix)
                isequal(merged_cov_trfs[target_ix], cov_var_srcs) ||
                    throw(ArgumentError(
                        "Conflicting covariance transforms for parameter :$param"))
            else
                merged_cov_trfs[target_ix] = cov_var_srcs
            end
        end
    end

    target_cov_par_ixs = sort!(Int[
        target_ix for (target_ix, variance_sources) in merged_cov_trfs
        if !isnothing(variance_sources)
    ])
    if isempty(target_cov_par_ixs) && all(==(TransformVariables.asℝ), scalar_transforms)
        return nothing
    else
        cov_trfs = CovarianceTransforms(
            target_cov_par_ixs,
            Tuple{Int, Int, Real, Real}[
                merged_cov_trfs[cov_ix] for cov_ix in target_cov_par_ixs],
            length(target_params)
        )
        return ParamTransforms(target_params, scalar_transforms, cov_trfs)
    end
end

"""
    var1_var2_covscale(model_vals, variance_source)

Return the two covariance-source variances and their geometric-mean covariance
scale. Each zero source index selects the corresponding fixed value stored in
`variance_source`.
"""
@inline function var1_var2_covscale(
    model_vals, variance_source::Tuple{Int, Int, T, T}
) where {T <: Real}
    var1_ix, var2_ix, fixed_var1, fixed_var2 = variance_source
    var1 = iszero(var1_ix) ? fixed_var1 : model_vals[var1_ix]
    var2 = iszero(var2_ix) ? fixed_var2 : model_vals[var2_ix]
    (isfinite(var1) && var1 > 0) || throw(DomainError(
        var1, "Covariance-source variances must be positive and finite"))
    (isfinite(var2) && var2 > 0) || throw(DomainError(
        var2, "Covariance-source variances must be positive and finite"))
    covscale = sqrt(var1) * sqrt(var2)
    isfinite(covscale) || throw(DomainError(
        (var1, var2), "Covariance scale must be finite"))
    return var1, var2, covscale
end

function _transform_param_group!(
    model_vals, scalar_derivatives, group, unconstrained_vals
)
    if isnothing(scalar_derivatives)
        @inbounds for (i, transform) in zip(group.indices, group.transforms)
            model_vals[i] =
                TransformVariables.transform(transform, unconstrained_vals[i])
        end
    else
        @inbounds for (i, transform, flip) in
                      zip(group.indices, group.transforms, group.flips)
            val, logabsderiv = TransformVariables.transform_and_logjac(
                transform, unconstrained_vals[i])
            model_vals[i] = val
            abs_deriv = exp(logabsderiv)
            scalar_derivatives[i] = ifelse(flip, -abs_deriv, abs_deriv)
        end
    end
    return model_vals
end

"""
    transform_params!(
        model_vals,
        scalar_derivatives,
        transforms,
        unconstrained_vals,
    )

Transform `unconstrained_vals` into model-parameter space, writing the result to
`model_vals` and returning `model_vals`.

When `scalar_derivatives` is a vector, write the diagonal derivatives of the scalar
transforms to it. Pass `nothing` when derivatives are not needed. These derivatives
exclude covariance scaling; [`pullback_param_gradient!`](@ref) applies that coupling
separately.

This mutating interface is available to optimizer extensions that need reusable
destination arrays.
"""
function transform_params!(
    model_vals::AbstractVector,
    scalar_derivatives::Union{AbstractVector, Nothing},
    transforms::ParamTransforms,
    unconstrained_vals::AbstractVector,
)
    check_params_vector(model_vals, transforms, unconstrained_vals)
    if !isnothing(scalar_derivatives)
        length(scalar_derivatives) == length(unconstrained_vals) ||
            throw(DimensionMismatch(
                "Derivative and parameter vectors have different lengths"))
    end
    foreach(transforms.groups) do group
        _transform_param_group!(
            model_vals, scalar_derivatives, group, unconstrained_vals)
    end
    _scale_covariances!(model_vals, transforms.covariance_transforms)
    return model_vals
end

"""
    transform_params(transforms, unconstrained_vals)

Return `unconstrained_vals` transformed into the model-parameter space described by
`transforms`. Use [`transform_params!`](@ref) when scalar derivatives or a reusable
destination vector are needed.
"""
transform_params(
    transforms::ParamTransforms, unconstrained_vals::AbstractVector) =
    transform_params!(
        similar(unconstrained_vals), nothing, transforms, unconstrained_vals)

function _inverse_transform_param_group!(
    unconstrained_vals, scalar_derivatives, group)
    @inbounds for (i, transform, flip) in
                  zip(group.indices, group.transforms, group.flips)
        value = unconstrained_vals[i]
        transformed = try
            TransformVariables.inverse(transform, value)
        catch error
            if error isa DomainError || error isa ArgumentError
                throw(DomainError(value,
                    "Starting model value for parameter index $i is outside " *
                    "the transformation domain"))
            end
            rethrow()
        end
        isfinite(transformed) || throw(DomainError(value,
            "Starting model value for parameter index $i maps to a non-finite " *
            "unconstrained value"))
        unconstrained_vals[i] = transformed
        if !isnothing(scalar_derivatives)
            _, logabsderiv = TransformVariables.transform_and_logjac(
                transform, transformed)
            abs_deriv = exp(-logabsderiv)
            scalar_derivatives[i] = ifelse(flip, -abs_deriv, abs_deriv)
        end
    end
    return unconstrained_vals
end

"""
    inverse_transform_params!(
        unconstrained_vals,
        scalar_derivatives,
        transforms,
        model_vals,
    )

Inverse-transform `model_vals`, writing the result to `unconstrained_vals` and
returning `unconstrained_vals`.

When `scalar_derivatives` is a vector, write the diagonal derivatives of the inverse
scalar transforms to it. Pass `nothing` when derivatives are not needed. These
derivatives exclude covariance unscaling.
"""
function inverse_transform_params!(
    unconstrained_vals::AbstractVector,
    scalar_derivatives::Union{AbstractVector, Nothing},
    transforms::ParamTransforms,
    model_vals::AbstractVector,
)
    check_params_vector(unconstrained_vals, transforms, model_vals)
    if !isnothing(scalar_derivatives)
        length(scalar_derivatives) == length(model_vals) ||
            throw(DimensionMismatch(
                "Derivative and parameter vectors have different lengths"))
    end
    copyto!(unconstrained_vals, model_vals)
    _unscale_covariances!(
        unconstrained_vals, transforms.covariance_transforms, model_vals)
    foreach(transforms.groups) do group
        _inverse_transform_param_group!(
            unconstrained_vals, scalar_derivatives, group)
    end
    return unconstrained_vals
end

"""
    inverse_transform_params(transforms, model_vals)

Return `model_vals` inverse-transformed into unconstrained parameter space. Use
[`inverse_transform_params!`](@ref) when inverse scalar derivatives or a reusable
destination vector are needed.
"""
inverse_transform_params(
    transforms::ParamTransforms, model_vals::AbstractVector) =
    inverse_transform_params!(
        similar(model_vals, float(eltype(model_vals))), nothing,
        transforms, model_vals)

"""
    pullback_param_gradient!(
        unconstrained_gradient,
        model_gradient,
        model_vals,
        scalar_derivatives,
        transforms,
    )

Pull a gradient in model-parameter space back to unconstrained space. This applies
both the covariance-scaling Jacobian and the scalar derivatives produced by
[`transform_params!`](@ref).

Optimizer extensions can use this after evaluating a model-space objective and gradient.
"""
function pullback_param_gradient!(
    unconstrained_grad::AbstractVector,
    model_grad::AbstractVector,
    model_vals::AbstractVector,
    scalar_derivatives::AbstractVector,
    transforms::ParamTransforms
)
    copyto!(unconstrained_grad, model_grad)
    cov_trfs = transforms.covariance_transforms
    @inbounds for (cov_ix, var_src) in zip(cov_trfs.covariance_indices, cov_trfs.variance_sources)
        var1_ix, var2_ix = var_src[1], var_src[2]
        var1, var2, covscale = var1_var2_covscale(model_vals, var_src)
        cov_val = model_vals[cov_ix]
        cov_grad = model_grad[cov_ix]
        unconstrained_grad[cov_ix] = cov_grad * covscale
        iszero(var1_ix) ||
            (unconstrained_grad[var1_ix] += cov_grad * cov_val / (2 * var1))
        iszero(var2_ix) ||
            (unconstrained_grad[var2_ix] += cov_grad * cov_val / (2 * var2))
    end
    unconstrained_grad .*= scalar_derivatives
    return unconstrained_grad
end

"""
    is_nonneg_transform(transform)

Return whether the scalar `transform` maps the extended real line to nonnegative
values. This is useful when deciding whether a separate nonnegativity constraint or
penalty would be redundant.
"""
function is_nonneg_transform(trf)
    range = try
        (
            TransformVariables.transform(trf, -Inf),
            TransformVariables.transform(trf, Inf),
        )
    catch
        nothing
    end
    return !isnothing(range) &&
        (range[1] isa Number) && (range[1] >= 0) &&
        (range[2] isa Number) && (range[2] >= 0)
end
