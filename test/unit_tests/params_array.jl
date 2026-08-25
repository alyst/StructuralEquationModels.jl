using Test
using LinearAlgebra
using StructuralEquationModels

const SEM = StructuralEquationModels
const TV = SEM.TransformVariables

@testset "reorder_params pins ParamsArray cells without rematerializing" begin
    old_params = [:var1, :var2, :cov]
    arr = SEM.ParamsMatrix{Float64}(
        Union{Float64, Symbol}[
            :var1 :cov
            :cov :var2
        ],
        old_params,
    )
    new_params = [:var1, :cov]
    replaced = SEM.reorder_params(arr, old_params, new_params, Dict(:var2 => 4.0))

    @test SEM.nparams(replaced) == 2
    @test collect(SEM.param_occurences(replaced, 1)) == collect(SEM.param_occurences(arr, 1))
    @test collect(SEM.param_occurences(replaced, 2)) == collect(SEM.param_occurences(arr, 3))
    materialized = SEM.materialize(replaced, new_params)
    @test materialized[1, 1] === :var1
    @test materialized[2, 2] == 4.0
    @test materialized[1, 2] === :cov
    @test materialized[2, 1] === :cov
end

@testset "reorder_params pins a RAM covariance source to a constant" begin
    parameters = [:var1, :var2, :cov]
    ram = SEM.RAMMatrices(
        A = zeros(2, 2),
        S = Union{Float64, Symbol}[
            :var1 :cov
            :cov :var2
        ],
        F = Matrix{Float64}(I, 2, 2),
        params = parameters,
        colnames = [:y1, :y2],
        param_transforms = SEM.ParamTransforms(
            parameters,
            [TV.asℝ₊, TV.asℝ₊, TV.as(Real, -1.0, 1.0) ∘ TV.TVScale(2.0)],
            SEM.CovarianceTransforms(
                [3],
                [(1, 2, 0.0, 0.0)],
                length(parameters),
            ),
        ),
    )
    pinned = SEM.reorder_params(ram, [:var1, :cov], Dict(:var2 => 4.0))

    @test SEM.params(pinned) == [:var1, :cov]
    @test SEM.materialize(pinned.S, SEM.params(pinned))[2, 2] == 4.0
    trfs = SEM.param_transforms(pinned)
    @test trfs.covariance_transforms.covariance_indices == [2]
    @test trfs.covariance_transforms.variance_sources == [(1, 0, 0.0, 4.0)]
end
