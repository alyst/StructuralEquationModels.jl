
vars(imply::SemImply) = vars(imply.ram_matrices)
observed_vars(imply::SemImply) = observed_vars(imply.ram_matrices)
latent_vars(imply::SemImply) = latent_vars(imply.ram_matrices)

nvars(imply::SemImply) = nvars(imply.ram_matrices)
nobserved_vars(imply::SemImply) = nobserved_vars(imply.ram_matrices)
nlatent_vars(imply::SemImply) = nlatent_vars(imply.ram_matrices)

params(imply::SemImply) = params(imply.ram_matrices)
nparams(imply::SemImply) = nparams(imply.ram_matrices)

param_transforms(imply::SemImply) = param_transforms(imply.ram_matrices)

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

# reset the covariation-related fields
function reset_covars!(imply::SemImply)
    imply._Σ_chol = nothing
    imply._isposdef_Σ = nothing
    imply._logdet_Σ = nothing
    imply._Σ⁻¹ = nothing
end

# update covariation-related fields
# some implementations may update some, but not all of the fields
# delaying the update of the specific fields until needed
function update_covars!(imply::SemImply)
    if isnothing(imply._Σ_chol) # skip if already updated
        BLAS.blascopy!(length(imply._Σ_chol_buf), _unwrap_symmetric(imply.Σ), 1, imply._Σ_chol_buf, 1)
        imply._Σ_chol = cholesky!(Symmetric(imply._Σ_chol_buf); check=false)
        imply._isposdef_Σ = isposdef(imply._Σ_chol)
        imply._logdet_Σ = imply._isposdef_Σ ? logdet(imply._Σ_chol) : NaN # cheap
    end
end

function isposdef_Σ(imply::SemImply)
    isnothing(imply._isposdef_Σ) && update_covars!(imply)
    return imply._isposdef_Σ
end

startswith_undescore(s::AbstractString) = startswith(s, '_')
startswith_undescore(s::Symbol) = startswith_undescore(String(s))

Base.propertynames(implied::SemImply, private::Bool=false) =
    (:logdet_Σ, :Σ⁻¹,
     (private ? fieldnames(typeof(implied)) :
                filter(!startswith_undescore, fieldnames(typeof(implied))))...)

@inline function Base.getproperty(imply::SemImply, name::Symbol)
    if name == :logdet_Σ # lazy log(det(Σ))
        isnothing(imply._logdet_Σ) && update_covars!(imply)
        return imply._logdet_Σ
    elseif name == :Σ⁻¹ # lazy Σ⁻¹
        if isnothing(imply._Σ⁻¹)
            update_covars!(imply)
            if isposdef_Σ(imply)
                imply._Σ⁻¹ = Symmetric(LinearAlgebra.inv!(imply._Σ_chol))
                imply._Σ_chol = nothing # invalidate Σ_chol since it got inverted
            else
                throw(LinearAlgebra.PosDefException(imply._Σ_chol.info))
            end
        end
        return imply._Σ⁻¹
    else
        return getfield(imply, name)
    end
end
