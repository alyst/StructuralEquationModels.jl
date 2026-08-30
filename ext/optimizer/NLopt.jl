############################################################################################
### connect to NLopt.jl as backend
############################################################################################

mutable struct NLoptResult
    result
    problem
end

SEM.optimizer(res::NLoptResult) = res.problem.algorithm
SEM.n_iterations(res::NLoptResult) = res.problem.numevals
SEM.convergence(res::NLoptResult) = res.result[3]

# construct SemFit from fitted NLopt object
function SemFit_NLopt(optimization_result, model::AbstractSem, start_params, opt)
    trfs = SEM.param_transforms(model)
    model_sol = isnothing(trfs) ? optimization_result[2] :
        SEM.transform_params(trfs, optimization_result[2])
    return SemFit(
        optimization_result[1],
        model_sol,
        start_params,
        model,
        NLoptResult(optimization_result, opt)
    )
end

# sem_fit method
function SEM.sem_fit(
    optim::SemOptimizerNLopt,
    model::AbstractSem,
    start_params::AbstractVector;
    kwargs...)

    model_trfs = SEM.param_transforms(model)
    if !isnothing(model_trfs)
        unconstrained_start_params = SEM.inverse_transform_params(
            model_trfs, start_params)
        _check_NLopt_transform_options(optim)
    else
        unconstrained_start_params = start_params
    end

    # construct the NLopt problem
    opt = construct_NLopt_problem(
        optim.algorithm,
        optim.options,
        length(unconstrained_start_params))
    set_NLopt_constraints!(opt, optim, model_trfs)
    if isnothing(model_trfs)
        opt.min_objective = (par, G) -> SEM.evaluate!(
            zero(eltype(par)), !isempty(G) ? G : nothing,
            nothing, model, par)
    else
        opt.min_objective = function (unconstrained_vals, unconstrained_gradient)
            return SEM.evaluate_unconstrained!(
                zero(eltype(unconstrained_vals)),
                !isempty(unconstrained_gradient) ? unconstrained_gradient : nothing,
                nothing, model, unconstrained_vals)
        end
    end

    if !isnothing(optim.local_algorithm)
        opt_local = construct_NLopt_problem(
            optim.local_algorithm,
            optim.local_options,
            length(unconstrained_start_params))
        opt.local_optimizer = opt_local
    end

    # fit
    result = NLopt.optimize(opt, unconstrained_start_params)

    return SemFit_NLopt(result, model, start_params, opt)
end

function _check_NLopt_transform_options(optimizer::SemOptimizerNLopt)
    for opt_options in (optimizer.options, optimizer.local_options)
        for bound in (:lower_bounds, :upper_bounds)
            haskey(opt_options, bound) && throw(ArgumentError(
                "NLopt $bound cannot be combined with parameter transforms; " *
                "use the transforms themselves or an explicit NLopt constraint"))
        end
    end
    return nothing
end

############################################################################################
### additional functions
############################################################################################

function construct_NLopt_problem(algorithm, options, npar)
    opt = Opt(algorithm, npar)

    for (key, val) in pairs(options)
        setproperty!(opt, key, val)
    end

    return opt

end

function set_NLopt_constraints!(
    opt::Opt,
    optimizer::SemOptimizerNLopt,
    transforms = nothing,
)
    for con in optimizer.inequality_constraints
        constraint = isnothing(transforms) ? con.f :
            _transformed_NLopt_constraint(con.f, transforms)
        inequality_constraint!(opt, constraint, con.tol)
    end
    for con in optimizer.equality_constraints
        constraint = isnothing(transforms) ? con.f :
            _transformed_NLopt_constraint(con.f, transforms)
        equality_constraint!(opt, constraint, con.tol)
    end
end

function _transformed_NLopt_constraint(constraint, transforms::SEM.ParamTransforms)
    return function (unconstrained_vals, unconstrained_gradient)
        model_vals = similar(unconstrained_vals, SEM.nparams(transforms))
        if isempty(unconstrained_gradient)
            SEM.transform_params!(
                model_vals, nothing, transforms, unconstrained_vals)
            return constraint(model_vals, unconstrained_gradient)
        end
        model_gradient = similar(model_vals)
        scalar_derivatives = similar(unconstrained_vals)
        SEM.transform_params!(
            model_vals, scalar_derivatives, transforms, unconstrained_vals)
        fill!(model_gradient, 0)
        result = constraint(model_vals, model_gradient)
        SEM.pullback_param_gradient!(
            unconstrained_gradient, model_gradient, model_vals,
            scalar_derivatives, transforms)
        return result
    end
end

############################################################################################
# pretty printing
############################################################################################

function Base.show(io::IO, result::NLoptResult)
    print(io, "Optimizer status: $(result.result[3]) \n")
    print(io, "Objective:        $(round(result.result[1]; digits = 2)) \n")
    print(io, "Algorithm:        $(result.problem.algorithm) \n")
    print(io, "No. evaluations:  $(result.problem.numevals) \n")
end
