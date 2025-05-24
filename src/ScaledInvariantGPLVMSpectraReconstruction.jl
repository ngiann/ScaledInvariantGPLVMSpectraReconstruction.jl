module ScaledInvariantGPLVMSpectraReconstruction

    using DifferentiationInterface
    using DispatchDoctor
    using Distances
    using Distributions
    using KernelFunctions
    using LinearAlgebra
    import Mooncake
    using Optim
    using Printf
    using Random
    import RectifierExpectations
    using StatsFuns
    using WoodburyMatrices
    
    
    using ForwardNeuralNetworks

    include("AbstractBackend.jl")

    include("initialscaleinvariantgplvmsolution.jl")
    include("initialmlscaleinvariantgplvmsolution.jl")
    include("calculatecovariance.jl")

    include("util/verify_bound_mlscaleinvariantgplvm.jl")
    include("util/verify_bound_gplvm.jl")
    include("util/woodbury.jl")
    include("util/default_JITTER.jl")
    include("util/quadratic_penalty.jl")
    include("util/marginalloglikelihood.jl")
    include("util/repeatoptimisation.jl")

    include("gaussianentropy.jl")
    include("expectation_log_prior.jl")

    include("scaleinvariantgplvm.jl")
    include("mlscaleinvariantgplvm.jl")
    include("numberofparameters_scaleinvariantgplvm.jl")
    include("numberofparameters_mlscaleinvariantgplvm.jl")
    include("mllowerbound.jl")
    include("lowerbound.jl")
    
    include("lowerbound_gplvm.jl")
    include("gplvm.jl")
    include("initialgplvmsolution.jl")
    include("numberofparameters_gplvm.jl")
    include("unpack_gplvm.jl")

    include("truncatednormal.jl")

    include("predict_scaleinvariantgplvm.jl")
    include("predict_gplvm.jl")
    
    export  gplvm, scaleinvariantgplvm, mlscaleinvariantgplvm, scaleinvariantgplvmpredictive, gplvmpredictive
    
    include("unpack_mlscaleinvariantgplvm.jl")
    include("unpack_scaleinvariantgplvm.jl")

    include("ppca.jl"); export ppca 
end