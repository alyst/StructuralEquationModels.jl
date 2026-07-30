TV = SEM.TransformVariables

correlation_transform = TV.as(Real, -1.0, 1.0) ∘ TV.TVScale(2.0)
scalar_transforms = [
    fill(TV.asℝ₊, 14)
    fill(correlation_transform, 6)
    fill(TV.asℝ, 11)
]
covariance_transforms = CovarianceTransforms(
    collect(15:20),
    [
        (4, 8, NaN, NaN), (5, 7, NaN, NaN),
        (5, 9, NaN, NaN), (6, 10, NaN, NaN),
        (7, 11, NaN, NaN), (9, 11, NaN, NaN),
    ],
    nparams(spec),
)
transformed_spec = RAMMatrices(
    spec;
    param_transforms = ParamTransforms(
        params(spec), scalar_transforms, covariance_transforms),
)

observed = SemObservedData(dat)
model_space_model = Sem(SemML(observed, RAM(spec)))
unconstrained_space_model = Sem(SemML(observed, RAM(transformed_spec)))
optimizer = if opt_engine == :Optim
    SemOptimizerOptim(
        algorithm = SEM.Optim.LBFGS(),
        options = SEM.Optim.Options(iterations = 2_000, g_tol = 1e-8),
    )
elseif opt_engine == :NLopt
    SemOptimizer(
        engine = :NLopt,
        algorithm = :LD_LBFGS,
        options = Dict(
            :maxeval => 2_000,
            :ftol_rel => 1e-12,
            :xtol_rel => 1e-10,
        ),
    )
else
    error("Unsupported optimizer engine: $opt_engine")
end

model_space_fit = sem_fit(optimizer, model_space_model, start_test)
unconstrained_space_fit = sem_fit(
    optimizer, unconstrained_space_model, start_test)

if opt_engine == :Optim
    @test SEM.convergence(model_space_fit)
    @test SEM.convergence(unconstrained_space_fit)
else
    successful_statuses = (:SUCCESS, :STOPVAL_REACHED, :FTOL_REACHED, :XTOL_REACHED)
    @test SEM.convergence(model_space_fit) in successful_statuses
    @test SEM.convergence(unconstrained_space_fit) in successful_statuses
end
@test SEM.minimum(unconstrained_space_fit) ≈
      SEM.minimum(model_space_fit) rtol = 1e-8
@test SEM.solution(unconstrained_space_fit) ≈
      SEM.solution(model_space_fit) rtol = 1e-4
@test SEM.objective(
          unconstrained_space_model, SEM.solution(unconstrained_space_fit)) ≈
      SEM.minimum(unconstrained_space_fit) rtol = 1e-12

backend_minimizer = if opt_engine == :Optim
    SEM.Optim.minimizer(unconstrained_space_fit.optimization_result)
else
    unconstrained_space_fit.optimization_result.result[2]
end
@test backend_minimizer ≈ SEM.inverse_transform_params(
    SEM.param_transforms(unconstrained_space_model),
    SEM.solution(unconstrained_space_fit),
) rtol = 1e-12

if opt_engine == :NLopt
    bounded_optimizer = SemOptimizer(
        engine = :NLopt,
        options = Dict(:lower_bounds => fill(-Inf, SEM.nparams(spec))),
    )
    @test_throws ArgumentError sem_fit(
        bounded_optimizer, unconstrained_space_model, start_test)

    model_values_seen = Ref(false)
    positive_variance_constraint = function (model_vals, model_gradient)
        model_values_seen[] |= all(>(0), model_vals[1:14])
        if !isempty(model_gradient)
            fill!(model_gradient, 0)
            model_gradient[1] = -1
        end
        return -model_vals[1]
    end
    constraint_optimizer = SemOptimizer(
        engine = :NLopt,
        algorithm = :LD_MMA,
        options = Dict(:maxeval => 2),
        inequality_constraints = (
            f = positive_variance_constraint,
            tol = 0.0,
        ),
    )
    sem_fit(constraint_optimizer, unconstrained_space_model, start_test)
    @test model_values_seen[]
end
