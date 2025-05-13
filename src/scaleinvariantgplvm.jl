function scaleinvariantgplvm(Y::Matrix{T}, σ::Matrix{T}; Q = 2, iterations = 100, seed = 1, H = 30, verify = false) where T<:Real

    rng = MersenneTwister(seed)

    D, N = size(Y)

    net = ThreeLayerNetwork(in = Q, H1 = H, H2 = H, out = D)
    
    p₀ = initialscaleinvariantgplvmsolution(rng, Q, N, net)

    scaleinvariantgplvm(Y, σ, p₀, net; Q = Q, iterations = iterations, verify = verify)

end


function scaleinvariantgplvm(Y::Matrix{T}, σ::Matrix{T}, p₀::Vector{T}, net; Q = 2, iterations = 1, verify = false) where T<:Real

    #------------------------------------------------------------
    # Check dimensions and preliminaries
    #------------------------------------------------------------

    D, N = size(Y); @assert(size(Y) == size(σ))

    numparam = numberofparameters_scaleinvariantgplvm(N, Q, net)
    
    @printf("siGPLVM: There are %d data items of dimension %d\n", N, D)
    @printf("Optimising %d number of free parameters\n", numparam)

    # Calculate observed diagonal covariance matrices

    S = σ.^2

    # verify bound
    
    # verify ? verify_bound_scaleinvariantgplvm(p₀; Y = Y, S = S, net = net, D = D, Q = Q, N = N) : nothing

    
    #------------------------------------------------------------
    # Setup and solve optimisation problem
    #------------------------------------------------------------

    helper(p) = -lowerbound(Y, S, unpack_scaleinvariantgplvm(p, net, Q, N)...)

    opt = Optim.Options(iterations = iterations, show_trace = true, show_every = 10)

    res = optimize(helper, p₀, ConjugateGradient(), opt, autodiff = AutoMooncake(config = nothing))


    return let

        local _mu,_λ, _β, X, ν, τ, w, _θ = unpack_scaleinvariantgplvm(res.minimizer, net, Q, N)

        local rec = net(w, X)

        c̄    = [g(νₙ, τₙ) for (νₙ, τₙ) in zip(ν, τ)]

        cvar = [v(νₙ, τₙ) for (νₙ, τₙ) in zip(ν, τ)]

        X, rec, res, net, res.minimum, c̄, cvar

    end
    
end




