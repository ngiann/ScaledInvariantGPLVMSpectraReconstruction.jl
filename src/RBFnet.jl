struct RBFnet{T}
    Φ::Matrix{T}
    Q::Int64
    M::Int64
    centres::Vector{T}
    t::Vector{T}
end


function create_rbfnet(;M = M, Q = Q, tobs = tobs, r = r)

    @assert(r > 0)

    t = (tobs .- minimum(tobs))

    t = t./maximum(t) * 2 .- 1

    centres = collect(LinRange(-1.0, 1.0, M))

    Φ = [exp(-0.5*abs2(tᵢ-c)/r)  for c in centres, tᵢ in t]

    # augment design matrix with bias row

    Φ = [Φ; ones(length(tobs))']

    RBFnet(Φ, Q, M+1, centres,t) # + 1 to account for bias in weight matrix

end

numparam(net::RBFnet) = net.Q*net.M

function (net::RBFnet)(param)

    Q, M = net.Q, net.M

    @assert(length(param) == Q*M)

    W = reshape(param[1:Q*M], Q, M)

    W*net.Φ

end


function get_t(net::RBFnet, W, r)

    t -> W*[[exp(-0.5*abs2(t-c)/r)  for c in net.centres]; 1.0]

end