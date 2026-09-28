using LinearAlgebra
using JuMP
using Mosek
using MosekTools
using DelimitedFiles
using SparseArrays

function bipartitions(n::Int)
    parts = Vector{Vector{Int}}()
    for m in 1:(2^(n-1)-1)
        push!(parts, reverse(digits(m, base=2, pad=n)))
    end
    return parts
end

function build_PT_permutation(Mvec, dims)
    n = length(dims)
    d = prod(dims)
    dim = dims[1]   # qubits assumed

    function lin_to_multi(k)
        digits(k-1, base=dim, pad=n)
    end

    function multi_to_lin(v)
        foldl((a,b)->a*dim+b, reverse(v)) + 1
    end

    rows = Int[]
    cols = Int[]

    for c in 1:d, r in 1:d
        i = lin_to_multi(r)
        j = lin_to_multi(c)

        ii = copy(i)
        jj = copy(j)

        for s in 1:n
            if Mvec[s] == 1
                ii[s], jj[s] = jj[s], ii[s]
            end
        end

        r′ = multi_to_lin(ii)
        c′ = multi_to_lin(jj)

        # vec index mapping (COLUMN MAJOR!)
        push!(rows, r′ + (c′-1)*d)
        push!(cols, r  + (c -1)*d)
    end

    return sparse(rows, cols, ones(length(rows)), d^2, d^2)
end


function partialtranspose(Q, PT, d)
    reshape(PT * vec(Q), d, d)
end

function build_all_PT_maps(dims)
    n = length(dims)
    PT_maps = Dict{Tuple{Vararg{Int}}, SparseMatrixCSC{Float64, Int}}()

    for Mvec in bipartitions(n)
        PT_maps[Tuple(Mvec)] = build_PT_permutation(Mvec, dims)
    end

    return PT_maps
end

function load_rho(filename)
    filename_r = "GMN_results/"*filename*"_r.txt"
    filename_i = "GMN_results/"*filename*"_i.txt"

    rho_r = readdlm(filename_r)
    rho_i = readdlm(filename_i)
    return rho_r + 1im*rho_i
end

function GMN(rho, dims, PT_maps)
    tol = 1e-8
    d = prod(dims)
    n = length(dims)

    @assert size(rho) == (d,d)
    @assert norm(rho - rho') < 1e-10

    model = Model(Mosek.Optimizer)

    @variable(model, W[1:d, 1:d], Hermitian)
    @constraint(model, tr(W)==1)

    for (Mtuple, PT) in PT_maps
        Qm = @variable(model, [1:d, 1:d] in HermitianPSDCone())

        Qm_vec = vec(Qm)
        Qm_PT_vec = PT*Qm_vec
        Qm_PT = reshape(Qm_PT_vec, d, d)

        @constraint(model, I(d)-Qm in HermitianPSDCone())
        @constraint(model, Hermitian(W-Qm_PT) in HermitianPSDCone())
    end

    @objective(model, Min, real(tr(rho*W)))

    optimize!(model)

    status = termination_status(model)
    minval = objective_value(model)

    if minval < -tol
        return minval, value.(W)
    else
        return minval, zeros(d, d)
    end
end

dims = [2,2,2,2,2,2]

PT_maps = build_all_PT_maps(dims)

GMN_val = zeros(9)
i = 1

ind1 = 0.0

#fname = "QSL Gapless_jd0.0049_"*string(j)*"_1_theta0.393_"

for j in 1:1
    #filename = "M1_"*string(j)*"_"*string(i)*"_theta"*string(ind1)
    filename="QSL"
              
    ρ = load_rho(filename)
    minval, Wopt = GMN(ρ, dims, PT_maps)
    GMN_val[j] = -minval
end

writedlm("GMN_calculation_res/GMN1_theta"*string(ind1)*".txt", GMN_val)

"""
dims = [2,2,2,2,2]

PT_maps = build_all_PT_maps(dims)

GMN_val = zeros(9)

for j in 1:9
    filename = "M2_"*string(j)*"_"*string(i)*"_theta"*string(ind1)
    ρ = load_rho(filename)
    minval, Wopt = GMN(ρ, dims, PT_maps)
    GMN_val[j] = -minval
end

writedlm("GMN_results/GMN2_theta"*string(ind1)*".txt", GMN_val)
"""
