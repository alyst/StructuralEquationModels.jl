using Test
using StructuralEquationModels
using LinearAlgebra
using FiniteDiff

const SEM = StructuralEquationModels
const Optim = SEM.Optim
const TV = SEM.TransformVariables

struct TransformQuadratic <: SEM.AbstractLoss{SEM.ExactHessian}
    target::Vector{Float64}
end

function SEM.evaluate!(objective, gradient, hessian,
                       loss::TransformQuadratic, parameters)
    residual = parameters .- loss.target
    isnothing(gradient) || copyto!(gradient, residual)
    if !isnothing(hessian)
        fill!(hessian, 0)
        hessian[diagind(hessian)] .= 1
    end
    return isnothing(objective) ? objective : sum(abs2, residual) / 2
end

params = [:variance, :covariance, :location]
target = [2.5, 0.4, -1.25]
model = SEM.Sem(TransformQuadratic(target); params)
transforms = Dict(
    :variance => TV.asℝ₊,
    :covariance => TV.as(Real, -1.0, 1.0),
)
prepared = SEM.ParamTransforms(params, transforms)

@testset "round trip and validation" begin
    unconstrained_vals = SEM.inverse_transform_params(prepared, target)
    @test SEM.transform_params(prepared, unconstrained_vals) ≈ target
    @test fieldnames(typeof(prepared)) ==
          (:transforms, :groups, :covariance_transforms)
    @test length(prepared.transforms) == length(params)
    @test all(group -> group.flips isa Vector{Bool}, prepared.groups)
    @test !SEM.allidentity(prepared)
    @test SEM.allidentity(SEM.ParamTransforms(
        params, fill(TV.asℝ, length(params))))
    @test !SEM.is_flipped(TV.asℝ)
    @test_throws DomainError SEM.inverse_transform_params(prepared, [0.0, 0.4, -1.25])
    @test_throws DomainError SEM.inverse_transform_params(prepared, [2.5, 1.0, -1.25])
    @test_throws ArgumentError SEM.ParamTransforms(params, Dict(:unknown => TV.asℝ₊))
    @test_throws DimensionMismatch SEM.ParamTransforms(params, [TV.asℝ])
end

@testset "chain rule" begin
    unconstrained_point = SEM.inverse_transform_params(prepared, [1.3, -0.2, 0.7])
    model_vals = similar(unconstrained_point)
    derivatives = similar(unconstrained_point)
    @test (SEM.transform_params!(
        model_vals, derivatives, prepared, unconstrained_point) === model_vals)
    recovered_unconstrained_vals = similar(unconstrained_point)
    inverse_derivatives = similar(unconstrained_point)
    @test (SEM.inverse_transform_params!(
        recovered_unconstrained_vals, inverse_derivatives,
        prepared, model_vals) === recovered_unconstrained_vals)
    @test recovered_unconstrained_vals ≈ unconstrained_point
    @test inverse_derivatives .* derivatives ≈ ones(length(derivatives))
    model_gradient = similar(unconstrained_point)
    SEM.evaluate!(nothing, model_gradient, nothing, model, model_vals)
    unconstrained_gradient = similar(unconstrained_point)
    SEM.pullback_param_gradient!(
        unconstrained_gradient, model_gradient, model_vals,
        derivatives, prepared)
    finite_difference_gradient = FiniteDiff.finite_difference_gradient(
        x -> SEM.objective(model, SEM.transform_params(prepared, x)),
        unconstrained_point,
    )
    @test unconstrained_gradient ≈ finite_difference_gradient rtol = 1e-6

end

@testset "batched covariance transforms" begin
    covariance_params = [
        :variance1, :variance2, :covariance12, :covariance1fixed, :location]
    covariance_target = [4.0, 9.0, 3.0, -4.0, 0.7]
    covariance_model = SEM.Sem(
        TransformQuadratic(covariance_target); params = covariance_params)
    correlation_transform = TV.as(Real, -1.0, 1.0) ∘ TV.TVScale(2.0)
    scalar_transforms = [
        TV.asℝ₊, TV.asℝ₊, correlation_transform,
        correlation_transform, TV.asℝ]
    covariance_transforms = SEM.CovarianceTransforms(
        [3, 4],
        [(1, 2, NaN, NaN), (1, 0, NaN, 16.0)],
        length(covariance_params),
    )
    coupled = SEM.ParamTransforms(
        covariance_params, scalar_transforms, covariance_transforms)
    @test SEM.var1_var2_covscale(
        covariance_target, covariance_transforms.variance_sources[1]) ==
        (4.0, 9.0, 6.0)
    @test SEM.var1_var2_covscale(
        covariance_target, covariance_transforms.variance_sources[2]) ==
        (4.0, 16.0, 8.0)

    unconstrained_vals = SEM.inverse_transform_params(coupled, covariance_target)
    @test unconstrained_vals[3] ≈ atanh(0.5)
    @test unconstrained_vals[4] ≈ atanh(-0.5)
    @test SEM.transform_params(coupled, unconstrained_vals) ≈ covariance_target
    transformed = SEM.transform_params(
        coupled, [log(4.0), log(9.0), 10.0, -10.0, 0.0])
    @test abs(transformed[3]) < sqrt(transformed[1] * transformed[2])
    @test abs(transformed[4]) < sqrt(transformed[1] * 16.0)
    @test abs(transformed[3]) > 1

    model_vals = similar(unconstrained_vals)
    scalar_derivatives = similar(unconstrained_vals)
    SEM.transform_params!(
        model_vals, scalar_derivatives, coupled, unconstrained_vals)
    model_gradient = similar(unconstrained_vals)
    SEM.evaluate!(
        nothing, model_gradient, nothing, covariance_model, model_vals)
    unconstrained_gradient = similar(unconstrained_vals)
    SEM.pullback_param_gradient!(
        unconstrained_gradient, model_gradient, model_vals,
        scalar_derivatives, coupled)
    finite_difference_gradient = FiniteDiff.finite_difference_gradient(
        x -> SEM.objective(
            covariance_model, SEM.transform_params(coupled, x)),
        unconstrained_vals .+ [0.1, -0.2, 0.15, -0.1, 0.2],
    )
    perturbed_unconstrained_vals =
        unconstrained_vals .+ [0.1, -0.2, 0.15, -0.1, 0.2]
    SEM.transform_params!(
        model_vals, scalar_derivatives, coupled, perturbed_unconstrained_vals)
    SEM.evaluate!(
        nothing, model_gradient, nothing, covariance_model, model_vals)
    SEM.pullback_param_gradient!(
        unconstrained_gradient, model_gradient, model_vals,
        scalar_derivatives, coupled)
    @test unconstrained_gradient ≈ finite_difference_gradient rtol = 1e-6

    @test_throws DomainError SEM.inverse_transform_params(
        coupled, [4.0, 9.0, 6.0, -4.0, 0.7])
    @test_throws ArgumentError SEM.CovarianceTransforms(
        [3], [(1, 0, NaN, 0.0)], length(covariance_params))
    converted = SEM.CovarianceTransforms(
        Int32[3],
        [(Int16(1), Int8(0), Float32(NaN), Float32(4))],
        length(covariance_params))
    @test converted.covariance_indices isa Vector{Int}
    @test converted.variance_sources isa
          Vector{Tuple{Int, Int, Float32, Float32}}
    @test_throws ArgumentError SEM.CovarianceTransforms(
        [0], [(1, 2, NaN, NaN)], length(covariance_params))
    @test_throws ArgumentError SEM.CovarianceTransforms(
        [3], [(-1, 2, NaN, NaN)], length(covariance_params))
    @test_throws ArgumentError SEM.CovarianceTransforms(
        [3], [(1, 6, NaN, NaN)], length(covariance_params))
    @test_throws ArgumentError SEM.CovarianceTransforms(
        [3], [(1, 3, NaN, NaN)], length(covariance_params))
    @test_throws ArgumentError SEM.CovarianceTransforms(
        [3, 3], [(1, 2, NaN, NaN), (1, 2, NaN, NaN)],
        length(covariance_params))
    @test_throws MethodError SEM.CovarianceTransforms(
        [3.0], [(1, 2, NaN, NaN)], length(covariance_params))
    @test_throws MethodError SEM.CovarianceTransforms(
        [3], [(1, 2, NaN, 1.0im)], length(covariance_params))

    symbolic_S = Union{Float64, Symbol}[
        :variance1 :covariance12 :covariance1fixed
        :covariance12 :variance2 0.0
        :covariance1fixed 0.0 16.0
    ]
    generated = SEM.CovarianceTransforms(symbolic_S, covariance_params)
    @test generated.covariance_indices == [3, 4]
    @test generated.variance_sources ==
          [(1, 2, 0.0, 0.0), (1, 0, 0.0, 16.0)]
    @test isempty(SEM.CovarianceTransforms(
        Union{Float64, Symbol}[:variance1 :variance1; :variance1 :variance1],
        covariance_params,
    ))
    @test_throws ArgumentError SEM.CovarianceTransforms(
        Union{Float64, Symbol}[:variance1 :covariance12; :covariance12 0.0],
        covariance_params,
    )

    reordered_params = [
        :location, :variance2, :unused, :variance1,
        :covariance1fixed, :covariance12]
    remapped = SEM.merge_param_transforms(
        reordered_params, [covariance_params => coupled])
    reordered_target = [0.7, 9.0, 2.0, 4.0, -4.0, 3.0]
    @test SEM.transform_params(
        remapped, SEM.inverse_transform_params(remapped, reordered_target)) ≈
        reordered_target

    positive = SEM.ParamTransforms([:variance], Dict(:variance => TV.asℝ₊))
    @test_throws ArgumentError SEM.merge_param_transforms(
        [:variance], [[:variance] => positive, [:variance] => nothing])

    merge_params = [:v1, :v2, :v3, :covariance]
    covariance12 = SEM.ParamTransforms(
        merge_params,
        fill(TV.asℝ, length(merge_params)),
        SEM.CovarianceTransforms(
            [4], [(1, 2, 0.0, 0.0)], length(merge_params)),
    )
    covariance21 = SEM.ParamTransforms(
        merge_params,
        fill(TV.asℝ, length(merge_params)),
        SEM.CovarianceTransforms(
            [4], [(2, 1, 0.0, 0.0)], length(merge_params)),
    )
    @test SEM.merge_param_transforms(
        merge_params,
        [merge_params => covariance12, merge_params => covariance21],
    ).covariance_transforms.variance_sources == [(1, 2, 0.0, 0.0)]
    covariance13 = SEM.ParamTransforms(
        merge_params,
        fill(TV.asℝ, length(merge_params)),
        SEM.CovarianceTransforms(
            [4], [(1, 3, 0.0, 0.0)], length(merge_params)),
    )
    @test_throws ArgumentError SEM.merge_param_transforms(
        merge_params,
        [merge_params => covariance12, merge_params => covariance13],
    )
end

@testset "model without transforms" begin
    optimizer = SEM.SemOptimizerOptim(
        algorithm = Optim.LBFGS(),
        options = Optim.Options(iterations = 200),
    )
    start = [0.3, -0.5, 2.0]
    fit = SEM.sem_fit(optimizer, model, start)
    @test isnothing(SEM.param_transforms(model))
    @test isnothing(SEM.param_transforms(SEM.ImplyEmpty(params)))
    @test fit.solution ≈ target atol = 1e-7
    @test Optim.minimizer(fit.optimization_result) == fit.solution
end


@testset "RAM likelihood" begin
    ram = SEM.RAMMatrices(
        A = zeros(2, 2),
        S = Union{Float64, Symbol}[:v1 0.0; 0.0 :v2],
        F = Matrix{Float64}(I, 2, 2),
        params = [:v1, :v2],
        colnames = [:y1, :y2],
        param_transforms = Dict(:v1 => TV.asℝ₊, :v2 => TV.asℝ₊),
    )
    observed = SEM.SemObservedCovariance(
        [2.0 0.0; 0.0 0.5]; n_obs = 100, specification = ram)
    ram_model = SEM.Sem(SEM.SemML(observed, SEM.RAM(ram)))
    optimizer = SEM.SemOptimizerOptim(
        algorithm = Optim.LBFGS(),
        options = Optim.Options(iterations = 500, g_tol = 1e-10),
    )
    @test SEM.param_transforms(ram) isa SEM.ParamTransforms
    @test SEM.param_transforms(ram_model) isa SEM.ParamTransforms
    @test SEM.start_simple(ram_model) == [1.0, 1.0]
    fit = SEM.sem_fit(optimizer, ram_model; start_params = SEM.start_simple)
    @test start_params(fit) == [1.0, 1.0]
    @test fit.solution ≈ [2.0, 0.5] atol = 1e-7
    @test SEM.objective(ram_model, fit.solution) ≈ fit.minimum atol = 1e-12

    unconstrained_vals = SEM.inverse_transform_params(
        SEM.param_transforms(ram_model), fit.solution)
    unconstrained_vals .+= [0.1, -0.2]
    unconstrained_gradient = similar(unconstrained_vals)
    unconstrained_objective = @inferred evaluate_unconstrained!(
        0.0, unconstrained_gradient, nothing, ram_model, unconstrained_vals)
    finite_difference_gradient = FiniteDiff.finite_difference_gradient(
        vals -> SEM.objective(
            ram_model,
            SEM.transform_params(SEM.param_transforms(ram_model), vals),
        ),
        unconstrained_vals,
    )
    @test unconstrained_objective ≈ SEM.objective(
        ram_model,
        SEM.transform_params(SEM.param_transforms(ram_model), unconstrained_vals),
    ) atol = 1e-12
    @test unconstrained_gradient ≈ finite_difference_gradient rtol = 1e-6
    @test Optim.minimizer(fit.optimization_result) ≈
          SEM.inverse_transform_params(SEM.param_transforms(ram_model), fit.solution)
    @test Optim.minimizer(fit.optimization_result) != fit.solution

    newton = SEM.SemOptimizerOptim(algorithm = Optim.Newton())
    @test_throws ArgumentError SEM.sem_fit(newton, ram_model, [1.0, 1.0])
    boxed = SEM.SemOptimizerOptim(algorithm = Optim.Fminbox(Optim.LBFGS()))
    @test_throws ArgumentError SEM.sem_fit(boxed, ram_model, [1.0, 1.0])

    expanded_ram = SEM.RAMMatrices(
        ram; params = [:v1, :v2, :unused_parameter])
    @test length(SEM.param_transforms(expanded_ram).transforms) == 3
    @test SEM.nonidentity_transformed_params(
        SEM.params(expanded_ram),
        SEM.param_transforms(expanded_ram)) == Set([:v1, :v2])
    @test isnothing(SEM.param_transforms(
        SEM.RAMMatrices(ram; param_transforms = nothing)))

    conflicting_ram = SEM.RAMMatrices(
        ram;
        params = SEM.params(ram),
        param_transforms = Dict(
            :v1 => TV.as(Real, -1.0, 1.0),
            :v2 => TV.asℝ₊,
        ),
    )
    conflicting_model = SEM.Sem(
        SEM.SemML(observed, SEM.RAM(ram)),
        SEM.SemML(observed, SEM.RAM(conflicting_ram)),
    )
    @test_throws ArgumentError SEM.param_transforms(conflicting_model)

    sparse_ram = SEM.RAMMatrices(
        A = Union{Float64, Symbol}[0.0 1.0; 0.0 0.0],
        S = Union{Float64, Symbol}[0.1 0.0; 0.0 :latent_variance],
        F = [1.0 0.0],
        params = [:latent_variance],
        colnames = [:y, :factor],
        param_transforms = Dict(:latent_variance => TV.asℝ₊),
    )
    sparse_observed = SEM.SemObservedCovariance(
        reshape([2.0], 1, 1); n_obs = 100, specification = sparse_ram)
    sparse_model = SEM.Sem(SEM.SemML(
        sparse_observed, SEM.RAMLargeSparse(sparse_ram)))
    sparse_fit = SEM.sem_fit(optimizer, sparse_model, [1.0])
    @test sparse_fit.solution ≈ [1.9] atol = 1e-7
    @test SEM.objective(sparse_model, sparse_fit.solution) ≈
          sparse_fit.minimum atol = 1e-12
end
