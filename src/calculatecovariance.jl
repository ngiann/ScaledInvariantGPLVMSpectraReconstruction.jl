@stable calculateK(X, θ::T) where T<:Real = kernelmatrix(with_lengthscale(SqExponentialKernel(), θ), X) + default_JITTER()*I

@stable calculateK(X, Y, θ::T) where T<:Real = kernelmatrix(with_lengthscale(SqExponentialKernel(), θ), X, Y)


@stable calculateK(X, θ::Vector{T}) where T<:Real = θ[1]*kernelmatrix(with_lengthscale(SqExponentialKernel(), θ[2]), X) + default_JITTER()*I

@stable calculateK(X, Y, θ::Vector{T}) where T<:Real = θ[1]*kernelmatrix(with_lengthscale(SqExponentialKernel(), θ[2]), X, Y)