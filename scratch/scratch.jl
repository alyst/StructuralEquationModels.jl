using Pkg
Pkg.activate(@__DIR__)

using Revise
using StructuralEquationModels, Test, LinearAlgebra

SEM = StructuralEquationModels

includet(
    joinpath(chop(dirname(pathof(StructuralEquationModels)), tail = 3),
    "test/examples/helper.jl")
    )

############################################################################################
### data
############################################################################################

dat = example_data("political_democracy")
dat_missing = example_data("political_democracy_missing")
solution_lav = example_data("political_democracy_solution")

############################################################################################
### specification - RAMMatrices
############################################################################################

# w.o. meanstructure -----------------------------------------------------------------------

x = Symbol.("x".*string.(1:31))

S =[:x1   0    0     0     0      0     0     0     0     0     0     0     0     0
    0     :x2  0     0     0      0     0     0     0     0     0     0     0     0
    0     0     :x3  0     0      0     0     0     0     0     0     0     0     0
    0     0     0     :x4  0      0     0     :x15  0     0     0     0     0     0
    0     0     0     0     :x5   0     :x16  0     :x17  0     0     0     0     0
    0     0     0     0     0     :x6  0      0     0     :x18  0     0     0     0
    0     0     0     0     :x16  0     :x7   0     0     0     :x19  0     0     0
    0     0     0     :x15 0      0     0     :x8   0     0     0     0     0     0
    0     0     0     0     :x17  0     0     0     :x9   0     :x20  0     0     0
    0     0     0     0     0     :x18 0      0     0     :x10  0     0     0     0
    0     0     0     0     0     0     :x19  0     :x20  0     :x11  0     0     0
    0     0     0     0     0     0     0     0     0     0     0     :x12  0     0
    0     0     0     0     0     0     0     0     0     0     0     0     :x13  0
    0     0     0     0     0     0     0     0     0     0     0     0     0     :x14]

F =[1.0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 1 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 1 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 1 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 1 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 1 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 1 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 1 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 1 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 1 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 1 0 0 0]

A =[0  0  0  0  0  0  0  0  0  0  0     1.0   0     0
    0  0  0  0  0  0  0  0  0  0  0     :x21  0     0
    0  0  0  0  0  0  0  0  0  0  0     :x22  0     0
    0  0  0  0  0  0  0  0  0  0  0     0     1.0   0
    0  0  0  0  0  0  0  0  0  0  0     0     :x23  0
    0  0  0  0  0  0  0  0  0  0  0     0     :x24  0
    0  0  0  0  0  0  0  0  0  0  0     0     :x25  0
    0  0  0  0  0  0  0  0  0  0  0     0     0     1
    0  0  0  0  0  0  0  0  0  0  0     0     0     :x26
    0  0  0  0  0  0  0  0  0  0  0     0     0     :x27
    0  0  0  0  0  0  0  0  0  0  0     0     0     :x28
    0  0  0  0  0  0  0  0  0  0  0     0     0     0
    0  0  0  0  0  0  0  0  0  0  0     :x29  0     0
    0  0  0  0  0  0  0  0  0  0  0     :x30  :x31  0]

spec = RAMMatrices(;
    A = A,
    S = S,
    F = F,
    parameters = x,
    colnames = [:x1, :x2, :x3, :y1, :y2, :y3, :y4, :y5, :y6, :y7, :y8, :ind60, :dem60, :dem65]
)

partable = ParameterTable(spec)

@test SEM.params(spec) == SEM.params(partable)

# w. meanstructure -------------------------------------------------------------------------

M = [:x32; :x33; :x34; :x35; :x36; :x37; :x38; :x35; :x36; :x37; :x38; 0.0; 0.0; 0.0]

spec_mean = RAMMatrices(;
    A = A,
    S = S,
    F = F,
    M = M,
    parameters = [spec.parameters; Symbol.("x", string.(32:38))],
    colnames = [:x1, :x2, :x3, :y1, :y2, :y3, :y4, :y5, :y6, :y7, :y8, :ind60, :dem60, :dem65])

partable_mean = ParameterTable(spec_mean)

solution_lav = example_data("political_democracy_solution")

lav_params = lav_parameters(solution_lav[:parameter_estimates_ml_mean], partable_mean, spec_mean.parameters)
lav_params = lav_parameters(solution_lav[:parameter_estimates_ls_mean], partable_mean)

model_ls = Sem(
    specification = spec_mean,
    data = dat,
    imply = RAMSymbolic,
    loss = SemWLS,
    meanstructure = true,
    optimizer = SemOptimizerOptim
)

model_ml = Sem(
    specification = spec_mean,
    #imply = RAMSymbolic,
    imply = RAM,
    data = dat,
    meanstructure = true,
    optimizer = SemOptimizerOptim
)

modelfit = sem_fit(model_ml)

obj_ml = SEM.objective!(model_ml, lav_params)
obj_ls = SEM.objective!(model_ls, lav_params)

lav_obj = solution_lav[:fitmeasures_ml][findfirst(==("fmin"), solution_lav[:fitmeasures_ml][:, 0x1]), 2]
lav_obj = solution_lav[:fitmeasures_ls][findfirst(==("fmin"), solution_lav[:fitmeasures_ls][:, 0x1]), 2]

lav_params = lav_parameters(solution_lav[:parameter_estimates_ml], partable)
lav_params = lav_parameters(solution_lav[:parameter_estimates_ls], partable)

model_ls = Sem(
    specification = spec,
    data = dat,
    imply = RAMSymbolic,
    loss = SemWLS,
    meanstructure = false,
    optimizer = SemOptimizerOptim
)

model_ml = Sem(
    specification = spec,
    imply = RAMSymbolic,
    #imply = RAM,
    data = dat,
    meanstructure = false,
    optimizer = SemOptimizerOptim
)

obj_ml = SEM.evaluate!(Float64, nothing, nothing, model_ml, lav_params)
obj_ls = SEM.evaluate!(Float64, nothing, nothing, model_ls, lav_params)

@test SEM.params(partable_mean) == SEM.params(spec_mean)

start_test = [fill(1.0, 11); fill(0.05, 3); fill(0.05, 6); fill(0.5, 8); fill(0.05, 3)]
start_test_mean = [fill(1.0, 11); fill(0.05, 3); fill(0.05, 6); fill(0.5, 8); fill(0.05, 3); fill(0.1, 7)]

semoptimizer = SemOptimizerOptim
@testset "RAMMatrices | constructor | Optim" begin include("constructor.jl") end

semoptimizer = SemOptimizerNLopt
@testset "RAMMatrices | constructor | NLopt" begin include("constructor.jl") end

if !haskey(ENV, "JULIA_EXTENDED_TESTS") || ENV["JULIA_EXTENDED_TESTS"] == "true"
    semoptimizer = SemOptimizerOptim
    @testset "RAMMatrices | parts | Optim" begin include("by_parts.jl") end
    semoptimizer = SemOptimizerNLopt
    @testset "RAMMatrices | parts | NLopt" begin include("by_parts.jl") end
end

@testset "constraints | NLopt" begin include("constraints.jl") end

############################################################################################
### specification - RAMMatrices → ParameterTable
############################################################################################

spec = ParameterTable(spec)
spec_mean = ParameterTable(spec_mean)

@test SEM.params(spec) == SEM.params(partable)

partable = spec
partable_mean = spec_mean

semoptimizer = SemOptimizerOptim
@testset "RAMMatrices → ParameterTable | constructor | Optim" begin include("constructor.jl") end
semoptimizer = SemOptimizerNLopt
@testset "RAMMatrices → ParameterTable | constructor | NLopt" begin include("constructor.jl") end

if !haskey(ENV, "JULIA_EXTENDED_TESTS") || ENV["JULIA_EXTENDED_TESTS"] == "true"
    semoptimizer = SemOptimizerOptim
    @testset "RAMMatrices → ParameterTable | parts | Optim" begin include("by_parts.jl") end
    semoptimizer = SemOptimizerNLopt
    @testset "RAMMatrices → ParameterTable | parts | NLopt" begin include("by_parts.jl") end
end

############################################################################################
### specification - Graph
############################################################################################

observed_vars = [:x1, :x2, :x3, :y1, :y2, :y3, :y4, :y5, :y6, :y7, :y8]
latent_vars = [:ind60, :dem60, :dem65]

graph = @StenoGraph begin
    # loadings
    ind60 → fixed(1)*x1 + x2 + x3
    dem60 → fixed(1)*y1 + y2 + y3 + y4
    dem65 → fixed(1)*y5 + y6 + y7 + y8
    # latent regressions
    label(:a)*dem60 ← ind60
    dem65 ← dem60
    dem65 ← ind60
    # variances
    _(observed_vars) ↔ _(observed_vars)
    _(latent_vars) ↔ _(latent_vars)
    # covariances
    y1 ↔ y5
    y2 ↔ y4 + y6
    y3 ↔ y7
    y8 ↔ y4 + y6
end

spec = ParameterTable(graph,
    latent_vars = latent_vars,
    observed_vars = observed_vars)

sort!(spec)

partable = spec

# meanstructure
mean_labels = label.([:m1, :m2, :m3, :m4, :m5, :m6, :m7, :m4, :m5, :m6, :m7])

graph = @StenoGraph begin
    # loadings
    ind60 → fixed(1)*x1 + x2 + x3
    dem60 → fixed(1)*y1 + y2 + y3 + y4
    dem65 → fixed(1)*y5 + y6 + y7 + y8
    # latent regressions
    label(:a)*dem60 ← ind60
    dem65 ← dem60
    dem65 ← ind60
    # variances
    _(observed_vars) ↔ _(observed_vars)
    _(latent_vars) ↔ _(latent_vars)
    # covariances
    y1 ↔ y5
    y2 ↔ y4 + y6
    y3 ↔ y7
    y8 ↔ y4 + y6
    # means
    Symbol("1") → _(mean_labels).*_(observed_vars)
    Symbol("1") → fixed(0)*ind60
end

spec_mean = ParameterTable(graph,
    latent_vars = latent_vars,
    observed_vars = observed_vars)

sort!(spec_mean)

partable_mean = spec_mean

start_test = [fill(0.5, 8); fill(0.05, 3); fill(1.0, 11);  fill(0.05, 9)]
start_test_mean = [fill(0.5, 8); fill(0.05, 3); fill(1.0, 11); fill(0.05, 3); fill(0.05, 13)]

semoptimizer = SemOptimizerOptim
@testset "Graph → ParameterTable | constructor | Optim" begin include("constructor.jl") end
semoptimizer = SemOptimizerNLopt
@testset "Graph → ParameterTable | constructor | NLopt" begin include("constructor.jl") end

if !haskey(ENV, "JULIA_EXTENDED_TESTS") || ENV["JULIA_EXTENDED_TESTS"] == "true"
    semoptimizer = SemOptimizerOptim
    @testset "Graph → ParameterTable | parts | Optim" begin include("by_parts.jl") end
    semoptimizer = SemOptimizerNLopt
    @testset "Graph → ParameterTable | parts | NLopt" begin include("by_parts.jl") end
end

model_g1 = Sem(
    specification = specification_g1,
    data = dat_g1,
    imply = RAMSymbolic
)

model_g2 = Sem(
    specification = specification_g2,
    data = dat_g2,
    imply = RAM
)

@test SEM.params(model_g1.imply.ram_matrices) == SEM.params(model_g2.imply.ram_matrices)

model_ml_multigroup = SemEnsemble(model_g1, model_g2; optimizer = semoptimizer)


# gradients
@testset "ml_gradients_multigroup" begin
    test_gradient(model_ml_multigroup, start_test; atol = 1e-9)
end

# fit
@testset "ml_solution_multigroup" begin
    solution = sem_fit(model_ml_multigroup)
    update_estimate!(partable, solution)
    test_estimates(
        partable,
        solution_lav[:parameter_estimates_ml]; atol = 1e-4,
        lav_groups = Dict(:Pasteur => 1, :Grant_White => 2))
end

@testset "fitmeasures/se_ml" begin
    solution_ml = sem_fit(model_ml_multigroup)
    test_fitmeasures(
        fit_measures(solution_ml),
        solution_lav[:fitmeasures_ml]; rtol = 1e-2, atol = 1e-7)

    update_partable!(
        partable, params(model_ml_multigroup), se_hessian(solution_ml), :se)
    test_estimates(
        partable,
        solution_lav[:parameter_estimates_ml]; atol = 1e-3,
        col = :se, lav_col = :se,
        lav_groups = Dict(:Pasteur => 1, :Grant_White => 2))
end

model_ml = Sem(
    specification = spec_mean,
    data = dat_missing,
    observed = SemObservedMissing,
    loss = SemFIML,
    optimizer = SemOptimizerOptim,
    meanstructure = true
)

fit = sem_fit(model_ml)

############################################################################################
# ML estimation - sorted
############################################################################################

partable_s = sort(partable)

specification_s = convert(Dict{Symbol, RAMMatrices}, partable_s)

specification_g1_s = specification_s[:Pasteur]
specification_g2_s = specification_s[:Grant_White]

dat = example_data("holzinger_swineford")
dat_missing = example_data("holzinger_swineford_missing")
solution_lav = example_data("holzinger_swineford_solution")

dat_g1 = filter(r -> r.school == "Pasteur", dat)
dat_g2 = dat[dat.school .== "Grant-White", :]

dat_miss_g1 = dat_missing[dat_missing.school .== "Pasteur", :]
dat_miss_g2 = dat_missing[dat_missing.school .== "Grant-White", :]

x = Symbol.(:x, 1:36)

F = zeros(9, 12)
F[diagind(F)] .= 1.0

A = Matrix{Any}(zeros(12, 12))
A[1, 10] = 1.0; A[4, 11] = 1.0; A[7, 12] = 1.0
A[2:3, 10] .= x[16:17]; A[5:6, 11] .= x[18:19]; A[8:9, 12] .= x[20:21];

# group 1
S1 = Matrix{Any}(zeros(12, 12))
S1[diagind(S1)] .= x[1:12]
S1[10, 11] = x[13]; S1[11, 10] = x[13]
S1[10, 12] = x[14]; S1[12, 10] = x[14]
S1[12, 11] = x[15]; S1[11, 12] = x[15]

# group 2
S2 = Matrix{Any}(zeros(12, 12))
S2[diagind(S2)] .= x[22:33]
S2[10, 11] = x[34]; S2[11, 10] = x[34]
S2[10, 12] = x[35]; S2[12, 10] = x[35]
S2[12, 11] = x[36]; S2[11, 12] = x[36]

specification_g1 = RAMMatrices(;
    A = A,
    S = S1,
    F = F,
    params = x,
    colnames = [:x1, :x2, :x3, :x4, :x5, :x6, :x7, :x8, :x9, :visual, :textual, :speed])

specification_g2 = RAMMatrices(;
    A = A,
    S = S2,
    F = F,
    params = x,
    colnames = [:x1, :x2, :x3, :x4, :x5, :x6, :x7, :x8, :x9, :visual, :textual, :speed])

partable = EnsembleParameterTable(
    Dict(:Pasteur => specification_g1,
         :Grant_White => specification_g2)
    )

partable_s = sort(partable)

specification_s = convert(Dict{Symbol, RAMMatrices}, partable_s)

specification_g1_s = specification_s[:Pasteur]
specification_g2_s = specification_s[:Grant_White]

model_g1 = Sem(
    specification = specification_g1_s,
    data = dat_g1,
    imply = RAMSymbolic
)

model_g2 = Sem(
    specification = specification_g2_s,
    data = dat_g2,
    imply = RAM
)

model_ls_sym = Sem(
    specification = spec,
    data = dat,
    imply = RAMSymbolic,
    loss = SemWLS,
    optimizer = SemOptimizerOptim
)

solution_ls = sem_fit(model_ls_sym)
fm = fit_measures(solution_ls)
@test all(test_fitmeasures(fm, solution_lav[:fitmeasures_ls]; atol = 1e-3,
    fitmeasure_names = fitmeasure_names_ls))
@test (fm[:AIC] === missing) & (fm[:BIC] === missing) & (fm[:minus2ll] === missing)

update_partable!(partable, identifier(model_ls_sym), se_hessian(solution_ls), :se)
compare_estimates(partable, solution_lav[:parameter_estimates_ls]; atol = 1e-2,
    col = :se, lav_col = :se)
