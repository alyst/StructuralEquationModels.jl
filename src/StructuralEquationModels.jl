module StructuralEquationModels

using LinearAlgebra, Optim,
    NLSolversBase, Statistics, StatsBase, SparseArrays, Symbolics,
    FiniteDiff, PrettyTables,
    Distributions, StenoGraphs, LazyArtifacts, DelimitedFiles,
    DataFrames,
    ProgressMeter, Printf,
    PackageExtensionCompat

export StenoGraphs, @StenoGraph, meld

const SEM = StructuralEquationModels

# type hierarchy
include("types.jl")
include("objective_gradient_hessian.jl")

# helper objects and functions
include("additional_functions/commutation_matrix.jl")

# fitted objects
include("frontend/fit/SemFit.jl")
# specification of models
include("additional_functions/params_array.jl")
include("frontend/specification/ParameterTable.jl")
include("frontend/specification/RAMMatrices.jl")
include("frontend/specification/EnsembleParameterTable.jl")
include("frontend/specification/StenoGraphs.jl")
include("frontend/fit/summary.jl")
include("frontend/predict.jl")
# pretty printing
include("frontend/pretty_printing.jl")
# observed
include("observed/prepare.jl")
include("observed/data.jl")
include("observed/covariance.jl")
include("observed/missing_pattern.jl")
include("observed/EM.jl")
include("observed/missing.jl")
# constructor
include("frontend/specification/Sem.jl")
include("frontend/specification/documentation.jl")
# imply
include("imply/abstract.jl")
include("imply/RAM/symbolic.jl")
include("imply/RAM/generic.jl")
include("imply/empty.jl")
# loss
include("loss/ML/abstract.jl")
include("loss/ML/ML.jl")
include("loss/ML/FIML.jl")
include("loss/WLS/WLS.jl")
include("loss/constant/constant.jl")

include("loss/regularization/lasso.jl")
include("loss/regularization/ridge.jl")
include("loss/regularization/hinge.jl")
include("loss/regularization/squared_hinge.jl")
# optimizer
include("diff/optim.jl")
include("diff/Empty.jl")
# optimizer
include("optimizer/documentation.jl")
include("optimizer/optim.jl")
# helper functions
include("additional_functions/helper.jl")
include("additional_functions/start_val/common.jl")
include("additional_functions/start_val/start_fabin3.jl")
include("additional_functions/start_val/start_partable.jl")
include("additional_functions/start_val/start_simple.jl")
include("additional_functions/artifacts.jl")
include("additional_functions/simulation.jl")
# fit measures
include("frontend/fit/fitmeasures/AIC.jl")
include("frontend/fit/fitmeasures/BIC.jl")
include("frontend/fit/fitmeasures/chi2.jl")
include("frontend/fit/fitmeasures/df.jl")
include("frontend/fit/fitmeasures/minus2ll.jl")
include("frontend/fit/fitmeasures/n_par.jl")
include("frontend/fit/fitmeasures/n_obs.jl")
include("frontend/fit/fitmeasures/p.jl")
include("frontend/fit/fitmeasures/RMSEA.jl")
include("frontend/fit/fitmeasures/n_man.jl")
include("frontend/fit/fitmeasures/fit_measures.jl")
# standard errors
include("frontend/fit/standard_errors/hessian.jl")
include("frontend/fit/standard_errors/bootstrap.jl")



export  ParamsArray, ParamsMatrix, ParamsVector,
            param_occurences, param_occurences_range,
            materialize, materialize!,
            sparse_gradient, sparse_gradient!, sparse_materialize,
        AbstractSem,
            Sem, SemFiniteDiff,
        MeanStructure, NoMeanStructure, HasMeanStructure,
        HessianEvaluation, ExactHessian, ApproximateHessian,
        SemImply,
            RAMSymbolic, RAMSymbolicZ, RAM, ImplyEmpty, imply,
        start_val,
            start_fabin3, start_simple, start_parameter_table,
        AbstractLoss,
            SemLoss, SemML, SemFIML, em_mvn,
            SemConstant, SemWLS, loss,
            SemLasso, SemRidge, SemHinge, SemSquaredHinge,
        SemOptimizer,
            SemOptimizerEmpty, SemOptimizerOptim,
            optimizer, n_iterations, convergence,
        SemObserved,
            SemObservedData, SemObservedCovariance, SemObservedMissing, observed,
        sem_fit,
        SemFit,
            minimum, solution,
        sem_summary,
        objective!, gradient!, hessian!, objective_gradient!, objective_hessian!,
            gradient_hessian!, objective_gradient_hessian!,
        SemSpecification,
            nparams, params,
            nlatent_vars, latent_vars,
            nobserved_vars, observed_vars,
            nvars, vars,
            variance_params,
        ParameterTable,
            EnsembleParameterTable, update_partable!, update_estimate!, update_start!, update_se_hessian!,
            Fixed, fixed, Start, start, Label, label, sort_vars!, sort_vars,
        RAMMatrices,
            RAMMatrices!,
        fit_measures,
            AIC, BIC, χ², df, fit_measures, minus2ll, n_obs, p_value, RMSEA, n_man,
            EmMVNModel,
        se_hessian, se_bootstrap,
        example_data,
        swap_observed, update_observed,
        @StenoGraph, →, ←, ↔, ⇔

function __init__()
    @require_extensions
end

end