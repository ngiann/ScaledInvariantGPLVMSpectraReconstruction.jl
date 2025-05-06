function verify_bound_gplvm(p₀; Y = Y, S = S, net = net, D = D, Q = Q, N = N)

    l1 = lowerbound_gplvm(Y, S, net, unpack_gplvm(p₀, net, Q, N)...)

    l2 = lowerbound_gplvm_slow(Y, S, net, unpack_gplvm(p₀, net, Q, N)...)
        
    @printf("Lower bound is      %.10f\n", l1)
    @printf("Alt lower bound is  %.10f\n", l2)
    @printf("Difference is       %.10f\n", l1 - l2)
end