"""
    start_simple(
        model;
        start_loadings = 0.5,
        start_regressions = 0.0,
        start_variances_observed = 1,
        start_variances_latent = 0.05,
        start_covariances_observed = 0.0,
        start_covariances_latent = 0.0,
        start_covariances_obs_lat = 0.0,
        start_means = 0.0,
        kwargs...)

Return a vector of simple starting values.
"""
function start_simple end

# Single Models ----------------------------------------------------------------------------
function start_simple(model::AbstractSemSingle; kwargs...)
    return start_simple(
        model.observed, 
        model.imply,
        model.loss.functions...,
        kwargs...)
end

function start_simple(observed, imply, args...; kwargs...)
    return start_simple(imply.ram_matrices; kwargs...)
end

# Ensemble Models --------------------------------------------------------------------------
function start_simple(model::SemEnsemble; kwargs...)
    
    start_vals = fill(0.0, nparams(model))

    for sem in model.sems
        sem_start_vals = start_simple(sem; kwargs...)
        for (i, val) in enumerate(sem_start_vals)
            if !iszero(val)
                start_vals[i] = val
            end
        end
    end

    return start_vals
end

function start_simple(
    ram_matrices::RAMMatrices;
    start_loadings = 0.5,
    start_regressions = 0.0,
    start_variances_observed = 1,
    start_variances_latent = 0.05,
    start_covariances_observed = 0.0,
    start_covariances_latent = 0.0,
    start_covariances_obs_lat = 0.0,
    start_means = 0.0,
    kwargs...
)
    A, S, F_ind, M, n_par =
        ram_matrices.A,
        ram_matrices.S,
        observed_var_indices(ram_matrices),
        ram_matrices.M,
        nparams(ram_matrices)

    start_val = zeros(n_par)
    n_obs = nobserved_vars(ram_matrices)
    n_var = nvars(ram_matrices)

    C_indices = CartesianIndices((n_var, n_var))

    for i in 1:n_par
        Si_ind = param_occurences(S, i)
        Ai_ind = param_occurences(A, i)
        if length(Si_ind) != 0
            # use the first occurence of the parameter to determine starting value
            c_ind = C_indices[Si_ind[1]]
            if c_ind[1] == c_ind[2]
                if c_ind[1] ∈ F_ind
                    start_val[i] = start_variances_observed
                else
                    start_val[i] = start_variances_latent
                end
            else
                o1 = c_ind[1] ∈ F_ind
                o2 = c_ind[2] ∈ F_ind
                if o1 & o2
                    start_val[i] = start_covariances_observed
                elseif !o1 & !o2
                    start_val[i] = start_covariances_latent
                else
                    start_val[i] = start_covariances_obs_lat
                end
            end
        elseif length(Ai_ind) != 0
            c_ind = C_indices[Ai_ind[1]]
            if (c_ind[1] ∈ F_ind) & !(c_ind[2] ∈ F_ind)
                start_val[i] = start_loadings
            else
                start_val[i] = start_regressions
            end
        elseif !isnothing(M) && (length(param_occurences(M, i)) != 0)
            start_val[i] = start_means
        end
    end
    return start_val
end