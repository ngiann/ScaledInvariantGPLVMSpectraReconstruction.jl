function initialmlscaleinvariantgplvmsolution(rng, Q, N, net)

    λ = ones(N)*100#rand(rng, N)

    β = 1.0
    
    X = randn(rng, Q*N)*1
    
    c = ones(N)#rand(rng, N)*3
    
    w = 0.1*randn(rng, ForwardNeuralNetworks.numweights(net))

    θ = 100.0

    [invsoftplus.(λ); invsoftplus(β); X; invsoftplus.(c); w; invsoftplus(θ)]

end