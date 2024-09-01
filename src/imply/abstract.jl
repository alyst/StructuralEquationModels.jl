
vars(imply::SemImply) = vars(imply.ram_matrices)
observed_vars(imply::SemImply) = observed_vars(imply.ram_matrices)
latent_vars(imply::SemImply) = latent_vars(imply.ram_matrices)

nvars(imply::SemImply) = nvars(imply.ram_matrices)
nobserved_vars(imply::SemImply) = nobserved_vars(imply.ram_matrices)
nlatent_vars(imply::SemImply) = nlatent_vars(imply.ram_matrices)

params(imply::SemImply) = params(imply.ram_matrices)
nparams(imply::SemImply) = nparams(imply.ram_matrices)

function check_acyclic(A::AbstractMatrix)
    # check if the model is acyclic
    acyclic = isone(det(I-A))

    # check if A is lower or upper triangular
    if istril(A)
        @info "A matrix is lower triangular"
        return LowerTriangular(A)
    elseif istriu(A)
        @info "A matrix is upper triangular"
        return UpperTriangular(A)
    else
        if acyclic
            @info "Your model is acyclic, specifying the A Matrix as either Upper or Lower Triangular can have great performance benefits.\n" maxlog=1
        end
        return A
    end
end

function reset_Σ_chol!(imply::SemImply)
    imply._Σ_chol = nothing
    imply._isposdef_Σ = nothing
    imply._logdet_Σ = nothing
    imply._Σ⁻¹ = nothing
    return nothing
end

function update_Σ_cholesky!(imply::SemImply)
    isnothing(imply._Σ_chol) || return nothing

    copy!(imply._Σ_chol_buf, imply.Σ)
    imply._Σ_chol = cholesky!(imply._Σ_chol_buf; check=false)
    imply._isposdef_Σ = isposdef(imply._Σ_chol)
    imply._logdet_Σ = imply._isposdef_Σ ? logdet(imply._Σ_chol) : NaN # cheap
    return nothing
end

function isposdef_Σ(imply::SemImply)
    isnothing(imply._isposdef_Σ) && update_Σ_cholesky!(imply)
    return imply._isposdef_Σ
end

@inline function Base.getproperty(imply::SemImply, name::Symbol)
    if name == :Σ_chol # lazy cholesky(Σ)
        isnothing(imply._Σ_chol) && update_Σ_cholesky!(imply)
        return imply._Σ_chol
    elseif name == :logdet_Σ # lazy log(det(Σ))
        isnothing(imply._logdet_Σ) && update_Σ_cholesky!(imply)
        return imply._logdet_Σ
    elseif name == :Σ⁻¹ # lazy Σ⁻¹
        if isnothing(imply._Σ⁻¹)
            imply._Σ⁻¹ = Symmetric(LinearAlgebra.inv!(imply.Σ_chol))
            imply._Σ_chol = nothing # invalidate Σ_chol since it got inverted
        end
        return imply._Σ⁻¹
    else
        return getfield(imply, name)
    end
end
