"""
    observed(loss::SemLoss) -> SemObserved

Returns the observed part of a model.
"""
observed(loss::SemLoss) = loss.observed

"""
    imply(loss::SemLoss) -> SemImply

Returns the imply part of a model.
"""
imply(loss::SemLoss) = loss.imply

for f in (:n_obs, :n_man, :obs_cov, :obs_mean)
    @eval $f(loss::SemLoss) = $f(observed(loss))
end

for f in (:vars, :nvars,
          :latent_vars, :nlatent_vars,
          :observed_vars, :nobserved_vars,
          :params, :nparams)
    @eval $f(loss::SemLoss) = $f(imply(loss))
end

#=
params(model::SemLoss) = params(imply(model))
nparams(model::SemLoss) = nparams(imply(model))

nvars(sem::SemLoss) = nvars(imply(sem))
nobserved_vars(sem::SemLoss) = nobserved_vars(imply(sem))
nlatent_vars(sem::SemLoss) = nlatent_vars(imply(sem))

vars(sem::SemLoss) = vars(imply(sem))
observed_vars(sem::SemLoss) = observed_vars(imply(sem))
latent_vars(sem::SemLoss) = latent_vars(imply(sem))

n_obs(sem::SemLoss) = n_obs(observed(sem))
n_man(sem::SemLoss) = n_man(observed(sem))
obs_cov(sem::SemLoss) = obs_cov(observed(sem))
obs_mean(sem::SemLoss) = obs_mean(observed(sem))
=#

function Base.show(io::IO, sem::SemLoss)
    println(io, "Structural Equation Model Loss ($(nameof(typeof(sem))))")
    println(io, "- Observed: $(nameof(typeof(observed(sem)))) ($(n_obs(sem)) observations)")
    println(io, "- Implied: $(nameof(typeof(imply(sem)))) ($(nparams(sem)) parameters)")
    println(io, "- Variables: $(nobserved_vars(sem)) observed, $(nlatent_vars(sem)) latent")
end

function check_observed_vars(observed::SemObserved, imply::SemImply)
    isnothing(observed_vars(imply)) ||
    observed_vars(observed) == observed_vars(imply) ||
        throw(ArgumentError("Observed variables defined for observed and imply do not match."))
end

check_observed_vars(sem::SemLoss) =
    check_observed_vars(observed(sem), imply(sem))
