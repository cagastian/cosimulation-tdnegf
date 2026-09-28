#!/usr/bin/env julia
#=
Bridge layer: exposes the NM ladder (square_electrons_lattice.jl) to the QSL
side with honeycomb site indexing, so the Python driver never has to think
about (col,row).

GEOMETRY REGISTRY (derived from bonds_kitaev in log_neg_fulldashboard_test.py)
--------------------------------------------------------------------------
The 14-site honeycomb cluster IS a 7x2 ladder:

    row 1 :  0 -x- 1 -y- 2 -x- 3 -y- 4 -x- 5 -y- 6      (cols 1..7)
             |z            |z            |z         |z
    row 2 : 13 -y- 12 -x- 11 -y- 10 -x-  9 -y-  8 -x- 7

    col c (1..7):   row 1 -> honeycomb site  c-1
                    row 2 -> honeycomb site  14-c

The four z-bonds (0,13),(2,11),(4,9),(6,7) are exactly the rungs at c=1,3,5,7.
NM linear index is l = (col-1)*Ny + row, matching src/hamiltonians.jl.

USAGE (from Python via juliacall):
    include("square_electrons_lattice.jl")
    include("qsl_nm_bridge.jl")
    dev = nm_setup()
    relax!(dev, 20.0)
    apply_S_honeycomb!(dev, S14, j_sd)   # S14 :: 14x3 Float64, honeycomb order
    step!(dev, dt)
    sig = spin_density_honeycomb(dev)    # 14x3, honeycomb order
    Ic, Is = lead_currents(dev)
=#

using StaticArrays

"Honeycomb site index (0-based, as in Python) for NM (col,row)."
hc_site(col::Int, row::Int) = row == 1 ? col - 1 : 14 - col

"""
    honeycomb_to_grid(S14) -> Matrix{SVector{3,Float64}}  (Nx x Ny)

`S14` is a 14x3 array in *honeycomb* (Python, 0-based) site order.
Returns the Nx x Ny grid of SVectors that `update_H_e!` expects.
Hard-coded to Nx=7, Ny=2 because that is the only geometry the registry
above is defined for -- it will throw if you change the cluster.
"""
function honeycomb_to_grid(S14::AbstractMatrix{<:Real})
    size(S14) == (14, 3) || throw(DimensionMismatch("S14 must be 14x3, got $(size(S14))"))
    G = Matrix{SVector{3,Float64}}(undef, 7, 2)
    for col in 1:7, row in 1:2
        i = hc_site(col, row) + 1            # Julia 1-based row into S14
        G[col, row] = SVector{3,Float64}(S14[i, 1], S14[i, 2], S14[i, 3])
    end
    return G
end

"""
    apply_S_honeycomb!(dev, S14, j_sd)

Rebuild H_TB from H0 plus the Kondo field using QSL expectation values given
in honeycomb site order.  Wraps `apply_sd_field!`.

SIGN: the package implements H = H0 - j_sd*(S.sigma).  See the note in
square_electrons_lattice.jl.  The Python driver owns the sign convention and
passes j_sd already signed -- do not add another minus here.
"""
function apply_S_honeycomb!(dev::NMDevice, S14::AbstractMatrix{<:Real}, j_sd::Real)
    apply_sd_field!(dev, honeycomb_to_grid(S14), Float64(j_sd))
    return dev
end

"""
    spin_density_honeycomb(dev) -> Matrix{Float64}  (14 x 3)

<sigma_j>(t) re-indexed into honeycomb order so row i corresponds to Python
site i (0-based).  Columns are (x,y,z).  Pauli-normalised (|.| <= 1).
"""
function spin_density_honeycomb(dev::NMDevice)
    sig = spin_density(dev)                  # (N_sites, 3) in NM linear order
    out = zeros(Float64, 14, 3)
    for col in 1:7, row in 1:2
        l = lin_idx(dev, col, row)
        out[hc_site(col, row) + 1, :] = sig[l, :]
    end
    return out
end

"""
    nm_readout(dev) -> (t, sigma14, Ic, Is)

One call that returns everything the Python driver needs after a step:
current time, <sigma> in honeycomb order (14x3), charge current per lead (2,)
and spin current per lead (2x3).  Bundled to keep the juliacall round-trip
count down -- each crossing has non-trivial overhead.
"""
function nm_readout(dev::NMDevice)
    Ic, Is = lead_currents(dev)
    return (current_time(dev), spin_density_honeycomb(dev), Ic, Is)
end

"""
    check_registry()

Verify the honeycomb<->ladder map reproduces the Kitaev bond table, i.e. that
every Kitaev bond is a nearest-neighbour pair on the 7x2 grid.  Cheap sanity
test -- run it once after any change to the cluster.
"""
function check_registry()
    bonds = [(0,1),(2,3),(4,5),(7,8),(9,10),(11,12),
             (1,2),(3,4),(5,6),(8,9),(10,11),(12,13),
             (0,13),(2,11),(4,9),(6,7)]
    pos = Dict{Int,Tuple{Int,Int}}()
    for col in 1:7, row in 1:2
        pos[hc_site(col,row)] = (col,row)
    end
    ok = true
    for (i,j) in bonds
        (ci,ri) = pos[i]; (cj,rj) = pos[j]
        d = abs(ci-cj) + abs(ri-rj)
        if d != 1
            @warn "bond ($i,$j) maps to non-adjacent grid sites" pos_i=(ci,ri) pos_j=(cj,rj)
            ok = false
        end
    end
    ok && @info "registry OK: all 16 Kitaev bonds are grid nearest neighbours"
    return ok
end
