############################################################################################
# struct
############################################################################################
"""
    SemFit

Fitted structural equation model.

# Interfaces
- `minimum(::SemFit)` -> minimum objective value
- `solution(::SemFit)` -> parameter estimates
- `start_params(::SemFit)` -> starting parameter values
- `model(::SemFit)`
- `optimization_result(::SemFit)`

- `optimizer(::SemFit)` -> optimization algorithm
- `n_iterations(::SemFit)` -> number of iterations
- `convergence(::SemFit)` -> convergence properties
"""
mutable struct SemFit{Mi, So, St, Mo, O}
    minimum::Mi
    solution::So
    start_params::St
    model::Mo
    optimization_result::O
end

params(fit::SemFit) = params(fit.model)

############################################################################################
# pretty printing
############################################################################################

function Base.show(io::IO, semfit::SemFit)
    println(io, "Fitted Structural Equation Model")
    println(io, "===============================================")
    println(io, "- $(nparams(semfit)) parameters")
    println(io)
    #print(io, "Objective value: $(round(semfit.minimum, digits = 4)) \n")
    println(io, "------------- Optimization result -------------")
    println(io)
    println(io, semfit.optimization_result)
    println(io, "------------- SEM Term Objectives -------------")
    for term in loss_terms(semfit.model)
        if !isnothing(id(term))
            print(io, ":$(id(term)): ")
        end
        print(io, nameof(losstype(term)))
        termobj = objective(loss(term), semfit.solution)
        @printf(io, " f=%.6g", termobj)
        if !isnothing(weight(term))
            @printf(io, " w=%.3g w*f=%.6g",
                    weight(term), weight(term) * termobj)
        else
            print(io, " w=1")
        end
        println(io)
    end
end

############################################################################################
# additional methods
############################################################################################

# access fields
minimum(sem_fit::SemFit) = sem_fit.minimum
solution(sem_fit::SemFit) = sem_fit.solution
start_params(sem_fit::SemFit) = sem_fit.start_params
model(sem_fit::SemFit) = sem_fit.model
optimization_result(sem_fit::SemFit) = sem_fit.optimization_result

# optimizer properties
optimizer(sem_fit::SemFit) = optimizer(optimization_result(sem_fit))
n_iterations(sem_fit::SemFit) = n_iterations(optimization_result(sem_fit))
convergence(sem_fit::SemFit) = convergence(optimization_result(sem_fit))
