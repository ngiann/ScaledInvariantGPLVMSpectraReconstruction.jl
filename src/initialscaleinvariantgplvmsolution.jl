function initialscaleinvariantgplvmsolution(rng, Q, N, net)

    λ = ones(N)*100#rand(rng, N)

    β = 1.0
    
    ν = ones(N)

    τ = ones(N)
    
    X = randn(rng, Q*N)*1
    
    w = 0.1*randn(rng, ForwardNeuralNetworks.numweights(net))

    θ = 100.0

    [invsoftplus.(λ); invsoftplus(β); ν; invsoftplus.(τ); X; w; invsoftplus(θ)]

end