using Test, LinearAlgebra
using StructuralEquationModels

const SEM = StructuralEquationModels

ram = SEM.RAMMatrices(
    A = [
        0.0 1.0 0.0
        0.0 0.0 1.0
        0.0 0.0 0.0
    ],
    S = Union{Float64, Symbol}[
        :residual_variance 0.0 0.0
        0.0 0.0 0.0
        0.0 0.0 :latent_variance
    ],
    F = [1.0 0.0 0.0],
    params = [:residual_variance, :latent_variance],
    colnames = [:y, :dependent_factor, :active_factor],
)
model_vals = [0.4, 1.6]
targets = SEM.EvaluationTargets(0.0, nothing, nothing)
dense = SEM.RAM(ram)
SEM.update!(targets, dense, model_vals)
dense_Σ = Matrix(dense.Σ)
dense_Σ⁻¹ = inv(dense_Σ)
dense_logdet_Σ = logdet(dense_Σ)

observed = SEM.SemObservedCovariance(
    reshape([2.3], 1, 1); n_obs = 100, specification = ram)
dense_model = SEM.Sem(SEM.SemML(observed, dense))
dense_gradient = similar(model_vals)
dense_objective = SEM.objective_gradient!(
    dense_gradient, dense_model, model_vals)

@testset "dependent latent variables using try_fast_Σ=$(try_fast_Σ)" for try_fast_Σ in (true, false)

    sparse = SEM.RAMLargeSparse(ram; simplify = false, try_fast_Σ)
    SEM.update!(targets, sparse, model_vals)

    @test sparse.active_lat_ixs == [2]
    @test !isnothing(sparse.I_All_parr)
    I_A⁻¹ll_test = fill(NaN, 2, 2)
    sparse.I_A⁻¹ll_eval!(I_A⁻¹ll_test, model_vals)
    @test I_A⁻¹ll_test[1, 1] == 1.0
    @test I_A⁻¹ll_test[1, 2] == 1.0
    @test isnan(I_A⁻¹ll_test[2, 1]) # structural zero is not emitted by build_function
    @test I_A⁻¹ll_test[2, 2] == 1.0
    @test size(sparse.I_A⁻¹ol_active) == (1, 1)
    @test size(sparse._Σ⁻¹lo_buf) == (1, 1)
    @test size(sparse._Σ⁻¹lo_buf2) == (1, 1)
    @test size(sparse._Σ⁻¹ll_buf) == (1, 1)
    @test Matrix(sparse.Σ) ≈ dense_Σ
    @test Matrix(sparse.Σ⁻¹) ≈ dense_Σ⁻¹
    @test sparse.logdet_Σ ≈ dense_logdet_Σ

    sparse_model = SEM.Sem(SEM.SemML(observed, sparse))
    sparse_gradient = similar(model_vals)
    path_counts = (sparse.n_fast, sparse.n_fast_failed, sparse.n_slow)
    sparse_objective = SEM.objective_gradient!(
        sparse_gradient, sparse_model, model_vals)

    @test sparse_objective ≈ dense_objective
    @test sparse_gradient ≈ dense_gradient
    expected_counts = try_fast_Σ ?
        (path_counts[1] + 1, path_counts[2], path_counts[3]) :
        (path_counts[1], path_counts[2], path_counts[3] + 1)
    @test (sparse.n_fast, sparse.n_fast_failed, sparse.n_slow) == expected_counts
end

@testset "fast precision access before covariance with observed regressions" begin
    regression_ram = SEM.RAMMatrices(
        A = Union{Float64, Symbol}[
            0.0 :observed_regression :loading_1
            0.0 0.0                  :loading_2
            0.0 0.0                   0.0
        ],
        S = Union{Float64, Symbol}[
            :residual_variance_1 0.0                  0.0
            0.0                  :residual_variance_2 0.0
            0.0                   0.0                 :latent_variance
        ],
        F = [
            1.0 0.0 0.0
            0.0 1.0 0.0
        ],
        params = [
            :observed_regression,
            :loading_1,
            :loading_2,
            :residual_variance_1,
            :residual_variance_2,
            :latent_variance,
        ],
        colnames = [:y1, :y2, :factor],
    )
    regression_values = [0.15, 0.4, -0.25, 0.7, 0.9, 1.2]

    dense = SEM.RAM(regression_ram)
    SEM.update!(targets, dense, regression_values)
    expected_Σ = Matrix(dense.Σ)
    expected_Σ⁻¹ = inv(expected_Σ)

    precision_first = SEM.RAMLargeSparse(regression_ram; simplify = false)
    SEM.update!(targets, precision_first, regression_values)
    @test Matrix(precision_first.Σ⁻¹) ≈ expected_Σ⁻¹
    # Computing Σ afterwards reuses its buffer without corrupting the cached
    # precision, which lives in a separate buffer.
    @test Matrix(precision_first.Σ) ≈ expected_Σ
    @test Matrix(precision_first.Σ⁻¹) ≈ expected_Σ⁻¹
    @test precision_first.n_fast == 1
    @test precision_first.n_fast_failed == 0
    @test precision_first.n_slow == 0

    covariance_first = SEM.RAMLargeSparse(regression_ram; simplify = false)
    SEM.update!(targets, covariance_first, regression_values)
    @test Matrix(covariance_first.Σ) ≈ expected_Σ
    @test Matrix(covariance_first.Σ⁻¹) ≈ expected_Σ⁻¹
end
