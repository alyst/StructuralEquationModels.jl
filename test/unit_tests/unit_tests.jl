using Test, SafeTestsets

@safetestset "Multithreading" begin include("multithreading.jl") end

@safetestset "SemObs" begin include("data_input_formats.jl") end

@safetestset "ParamsArray" begin include("params_array.jl") end

@safetestset "Param Transforms" begin include("param_transforms.jl") end

@safetestset "Predict Scores" begin include("predict_scores.jl") end

@safetestset "Sem Metadata" begin include("sem_metadata.jl") end

@safetestset "RAMLargeSparse" begin include("largesparse.jl") end
