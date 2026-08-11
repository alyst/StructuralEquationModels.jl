using StructuralEquationModels
const SEM = StructuralEquationModels

model = SEM.Sem(
    SEM.SemConstant(1.0);
    params = [:parameter],
    metadata = (source = :unit_test, config = (iterations = 2,)),
)

@test SEM.metadata(model) == (source = :unit_test, config = (iterations = 2,))
@test SEM.metadata(SEM.SemFiniteDiff(model)) == SEM.metadata(model)

model_without_metadata = SEM.Sem(SEM.SemConstant(1.0); params = [:parameter])
@test isempty(SEM.metadata(model_without_metadata))
