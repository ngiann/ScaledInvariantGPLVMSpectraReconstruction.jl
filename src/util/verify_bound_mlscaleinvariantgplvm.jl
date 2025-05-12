function verify_bound_mlscaleinvariantgplvm(p₀; Y = Y, S = S, net = net, D = D, Q = Q, N = N, backend = backend)

    l1 = mllowerbound(Y, S, backend, unpack_mlscaleinvariantgplvm(p₀, net, Q, N)...)
        
    l2 = mllowerbound_slow(Y, S, backend, unpack_mlscaleinvariantgplvm(p₀, net, Q, N)...)
        
    # l3  = marginalloglikelihood(Y, S, unpack_gplvm(p₀, net, D, Q, N)...)

    @printf("Lower bound is      %.10f\n", l1)
    @printf("Alt lower bound is  %.10f\n", l2)
    @printf("Difference is       %.10f\n", l1 - l2)
    # @printf("Marginal log likelihood is (should be higher) %f\n", l3)
end