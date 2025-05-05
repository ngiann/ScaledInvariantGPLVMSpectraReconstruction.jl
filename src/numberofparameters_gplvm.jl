function numberofparameters_gplvm(N, Q, net)

    N + 3 + Q*N + ForwardNeuralNetworks.numweights(net)

    # N because of the λ parameters for the posterior
    # 3 because of β and the two parameters for the covariance matrix K
    # Q×N because of the latent variables X
    # Finally, the weights of the neural network
end