losstype(::Type{<:LossTerm{L,W,I}}) where {L, W, I} = L
losstype(term::LossTerm) = losstype(typeof(term))
loss(term::LossTerm) = term.loss
weight(term::LossTerm) = term.weight
id(term::LossTerm) = term.id

issemloss(term::LossTerm) = isa(loss(term), SemLoss)

for f in (:imply, :observed,
          :n_obs, :observed_vars, :nobserved_vars,
          :vars, :nvars, :latent_vars, :nlatent_vars, :params, :nparams)
    @eval $f(term::LossTerm) = $f(loss(term))
end

function Base.show(io::IO, term::LossTerm)
    if !isnothing(id(term))
        print(io, ":$(id(term)): ")
    end
    print(io, nameof(losstype(term)))
    if issemloss(term)
        print(io, " ($(n_obs(term)) observations, $(nobserved_vars(term)) observed, $(nlatent_vars(term)) latent variables)")
    end
    if !isnothing(weight(term))
        @printf(io, " w=%.3g", weight(term))
    else
        print(io, " w=1")
    end
end

############################################################################################
# constructor for Sem types
############################################################################################

function Sem(loss_terms...;
             params::Union{Vector{Symbol}, Nothing} = nothing,
             default_sem_weights = :n_obs,
)
    default_sem_weights ∈ [:n_obs, :uniform, :one] ||
        throw(ArgumentError("Unsupported default_sem_weights=:$default_sem_weights"))
    # assemble a list of weighted losses and check params equality
    terms = Vector{LossTerm}()
    params = !isnothing(params) ? copy(params) : params
    has_sem_weights = false
    nsems = 0
    for inp_term in loss_terms
        if inp_term isa AbstractLoss
            term = inp_term
            term_w = nothing
            term_id = nothing
        elseif inp_term isa Pair
            if inp_term[1] isa AbstractLoss
                term, term_w = inp_term
                term_id = nothing
            elseif inp_term[2] isa AbstractLoss
                term_id, term = inp_term
                term_w = nothing
            elseif inp_term[2] isa Pair
                term_id, (term, term_w) = inp_term
                isa(term, AbstractLoss) ||
                    throw(ArgumentError("AbstractLoss expected as a second argument of a loss term double pair (id => loss => weight), $(nameof(typeof(term))) found"))
            end
        else
            throw(ArgumentError("[id =>] AbstractLoss [=> weight] expected as a loss term, $(nameof(typeof(inp_term))) found"))
        end

        if term isa SemLoss
            nsems += 1
            has_sem_weights |= !isnothing(term_w)
            # check integrity
            if isnothing(params)
                params = SEM.params(term)
            elseif params != SEM.params(term)
                # FIXME the suggestion might no longer be relevant, since ParTable also stores params order
                error("The parameters of your SEM models do not match.\n
Maybe you tried to specify models of an ensemble via ParameterTables.\n
In that case, you may use RAMMatrices instead.")
            end
            check_observed_vars(term)
        elseif !(term isa AbstractLoss)
            throw(ArgumentError("AbstractLoss term expected at $(length(terms)+1) position, $(nameof(typeof(term))) found"))
        end
        push!(terms, LossTerm(term, term_id, term_w))
    end
    isnothing(params) && throw(ErrorException("No SEM models provided."))

    if !has_sem_weights && nsems > 1
        # set the weights of SEMs in the ensemble
        if default_sem_weights == :n_obs
            # weight SEM by the number of observations
            nobs_total = sum(n_obs(term) for term in terms if issemloss(term))
            for (i, term) in enumerate(terms)
                if issemloss(term)
                    terms[i] = LossTerm(loss(term), id(term), n_obs(term) / nobs_total)
                end
            end
        elseif default_sem_weights == :uniform # uniform weights
            for (i, term) in enumerate(terms)
                if issemloss(term)
                    terms[i] = LossTerm(loss(term), id(term), 1/nsems)
                end
            end
        elseif default_sem_weights == :one # do nothing
        end
    end

    terms_tuple = Tuple(terms)
    return Sem{typeof(terms_tuple)}(terms_tuple, params)
end

############################################################################################
# functions
############################################################################################

params(model::AbstractSem) = model.params

loss_terms(model::AbstractSem) = model.loss_terms
nloss_terms(model::AbstractSem) = length(loss_terms(model))

"""
    sem_terms(sem::AbstractSem)

Returns a tuple of all [`SemLoss`](@ref) weighted terms in `sem`.
"""
sem_terms(sem::AbstractSem) = Tuple(term for term in loss_terms(sem) if issemloss(term))
nsem_terms(sem::AbstractSem) = sum(issemloss, loss_terms(sem))

n_obs(sem::AbstractSem) = sum(term -> issemloss(term) ? n_obs(term) : 0, loss_terms(sem))

function sem_term(model::AbstractSem)
    if nsem_terms(model) != 1
        error("Model contains $(nsem_terms(model)) SEM terms, you have to specify a specific term")
    end
    for term in loss_terms(sem)
        issemloss(term) && return loss(term)
    end
    error("Unreachable reached")
end

function loss_term(model::AbstractSem, id::Any)
    for term in loss_terms(model)
        if SEM.id(term) == id
            return loss(term)
        end
    end
    error("No loss term with id=$id found")
end

function set_field_type_kwargs!(kwargs, observed, imply, loss, O, I)
    kwargs[:observed_type] = O <: Type ? observed : typeof(observed)
    kwargs[:imply_type] = I <: Type ? imply : typeof(imply)
    if loss isa SemLoss
        kwargs[:loss_types] = [lossfun isa SemLossFunction ? typeof(lossfun) : lossfun for lossfun in loss.functions]
    elseif applicable(iterate, loss)
        kwargs[:loss_types] = [lossfun isa SemLossFunction ? typeof(lossfun) : lossfun for lossfun in loss]
    else
        kwargs[:loss_types] = [loss isa SemLossFunction ? typeof(loss) : loss]
    end
end

# construct Sem fields
function get_fields!(kwargs, observed, imply, loss)
    # observed
    if !isa(observed, SemObserved)
        observed = observed(;kwargs...)
    end
    kwargs[:observed] = observed

    # imply
    if !isa(imply, SemImply)
        imply = imply(;kwargs...)
    end

    kwargs[:imply] = imply
    kwargs[:nparams] = nparams(imply)

    # loss
    loss = get_SemLoss(loss; kwargs...)
    kwargs[:loss] = loss

    return observed, imply, loss
end

# construct loss field
function get_SemLoss(loss; kwargs...)
    if loss isa SemLoss
        nothing
    elseif applicable(iterate, loss)
        loss_out = []
        for lossfun in loss
            if isa(lossfun, SemLossFunction)
                push!(loss_out, lossfun)
            else
                lossfun = lossfun(;kwargs...)
                push!(loss_out, lossfun)
            end
        end
        loss = SemLoss(loss_out...; kwargs...)
    else
        if !isa(loss, SemLossFunction)
            loss = SemLoss(loss(;kwargs...); kwargs...)
        else
            loss = SemLoss(loss; kwargs...)
        end
    end
    return loss
end


##############################################################
# pretty printing
##############################################################

function Base.show(io::IO, sem::AbstractSem)
    println(io, "Structural Equation Model ($(nameof(typeof(sem))))")
    println(io, "- $(nparams(sem)) parameters")
    println(io, "- Loss terms:")
    for term in loss_terms(sem)
        print(io, "  - ")
        print(io, term)
        println(io)
    end
end
