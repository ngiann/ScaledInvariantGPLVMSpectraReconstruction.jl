function numberofparameters_mlscaleinvariantgplvm(N, Q, net)

    N + 2 + N + Q*N + ForwardNeuralNetworks.numweights(net)

    # N because of the λ parameters for the posterior
    # 2 because of β and the two parameters for the covariance matrix K
    # N because of the scaling parameters for each high dimensional data item
    # Q×N because of the latent variables X
    # Finally, the weights of the neural network
end
