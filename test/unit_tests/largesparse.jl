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
