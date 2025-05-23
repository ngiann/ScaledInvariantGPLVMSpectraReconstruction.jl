@stable function unpack_scaleinvariantgplvm(parameters, net, Q, N)

    @assert(numberofparameters_scaleinvariantgplvm(N, Q, net) == length(parameters))
    
    nwts = ForwardNeuralNetworks.numweights(net)

    MARK = 0

    λ = softplus.(parameters[MARK+1:MARK+N]); MARK += N

    β = softplus(parameters[MARK+1]); MARK += 1
    
    X = reshape(parameters[MARK+1:MARK+Q*N], Q, N); MARK += Q*N
    
    ν = (parameters[MARK+1:MARK+N]); MARK += N
    
    τ = softplus.(parameters[MARK+1:MARK+N]); MARK += N

    w = parameters[MARK+1:MARK+nwts]; MARK += nwts
    
    θ = softplus(parameters[MARK+1]); MARK += 1

    @assert(MARK == length(parameters)) # make sure we used up all parameters

    μ = net(w, X)

    return μ, λ, β, X, ν, τ, w, θ

end