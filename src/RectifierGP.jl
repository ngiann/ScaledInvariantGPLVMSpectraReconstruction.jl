module RectifierGP
    
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
    
    
    using ForwardNeuralNetworks

    include("AbstractBackend.jl")
    # include("loadmnist.jl")
    include("initialscaleinvariantgplvmsolution.jl")
    include("calculatecovariance.jl")
    # include("gp.jl")
    # include("rectifiergp.jl")
    include("util/verify_bound.jl")
    include("util/woodbury.jl")
    include("util/default_JITTER.jl")
    include("util/quadratic_penalty.jl")
    include("util/marginalloglikelihood.jl")
    include("util/repeatoptimisation.jl")

    include("gaussianentropy.jl")
    include("expectation_log_prior.jl")

    include("scaleinvariantgplvm.jl")
    include("numberofparameters_scaleinvariantgplvm.jl")
    include("lowerbound.jl")
    # include("grad/lowerbound_for_AD.jl")
    # include("grad/helper_grad_manual.jl")
    # include("grad/helper_grad_ad.jl")
    # include("grad/helper_grad.jl")

    include("lowerbound_gplvm.jl")
    include("gplvm.jl")
    include("initialgplvmsolution.jl")
    include("numberofparameters_gplvm.jl")
    include("unpack_gplvm.jl")


    include("predict_scaleinvariantgplvm.jl")
    include("predict_gplvm.jl")
    include("RBFnet.jl")
    
    export  create_rbfnet, gplvm, scaleinvariantgplvm, scaleinvariantgplvmpredictive, gplvmpredictive
    
    include("unpack_scaleinvariantgplvm.jl")

end
