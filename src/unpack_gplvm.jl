@stable function unpack_gplvm(parameters, net, Q, N)

    nwts = ForwardNeuralNetworks.numweights(net)

    MARK = 0

    λ = softplus.(parameters[MARK+1:MARK+N]); MARK += N

    β = softplus(parameters[MARK+1]); MARK += 1
    
    X = reshape(parameters[MARK+1:MARK+Q*N], Q, N); MARK += Q*N

    w = parameters[MARK+1:MARK+nwts]; MARK += nwts
    
    θ = softplus.(parameters[MARK+1:MARK+2]); MARK += 2

    @assert(MARK == length(parameters)) # make sure we used up all parameters

    return λ, β, X, w, θ

end