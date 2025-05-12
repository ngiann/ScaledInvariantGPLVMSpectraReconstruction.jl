function initialmlscaleinvariantgplvmsolution(rng, Q, N, net)

    λ = ones(N)*100#rand(rng, N)

    β = 1.0
    
    c = ones(N)#rand(rng, N)*3
    
    X = randn(rng, Q*N)*1
    
    w = 0.1*randn(rng, ForwardNeuralNetworks.numweights(net))

    θ = 100.0

    [invsoftplus.(λ); invsoftplus(β); invsoftplus.(c); X; w; invsoftplus(θ)]

end