abstract type AbstractBackend end

struct ExponentialBackend <: AbstractBackend

end

struct LinearBackend <: AbstractBackend

end

struct RecBackend <: AbstractBackend

end

# M stands for mean, V stands for variance

M(::RecBackend; a=a, b=b, μ=μ, σ=σ) = RectifierExpectations.M(a, b, μ, σ) 
V(::RecBackend; a=a, b=b, μ=μ, σ=σ) = RectifierExpectations.V(a, b, μ, σ)

M(::ExponentialBackend; a=a, b=b, μ=μ, σ=σ) = ExponentialExpectations.E(a = a, μ = μ, σ = σ, b = b)
V(::ExponentialBackend; a=a, b=b, μ=μ, σ=σ) = ExponentialExpectations.V(a = a, μ = μ, σ = σ, b = b)

M(::LinearBackend; a=a, b=b, μ=μ, σ=σ) = a*μ + b
V(::LinearBackend; a=a, b=b, μ=μ, σ=σ) = (abs(a)*σ)^2


_rectifier(x) = x > 0 ? x : zero(eltype(x))

transform(::RecBackend, x) = _rectifier.(x)

transform(::ExponentialBackend, x) = exp.(x)

transform(::LinearBackend, x) = x

