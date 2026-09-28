#!/usr/bin/env julia
#=
NM layer of the FI / QSL / NM stack — steppable TDNEGF driver.

This file sets up the normal-metal ladder (2 rows × 7 columns, two leads at the
SAME chemical potential μ) and exposes a *stepper* interface:

    dev = nm_setup()                      # build device, integrator at t = 0
    σ   = spin_density(dev)               # ⟨σ_j⟩ = Tr_spin[ρ_jj σ]  at current t
    set_hamiltonian!(dev, H_new)          # OR: apply_sd_field!(dev, S, J_s)
    step!(dev, Δt)                        # advance t → t + Δt with that H
    σ   = spin_density(dev)               # ⟨σ_j⟩ at the new t
    ...

No plotting, no solve-to-the-end. The integrator is held open so an external
driver (the QSL side) can interleave its own evolution of Ŝ_j.

Coupling term of the figure:   J_s ⟨Ŝ_j⟩ · σ̂_j
    ⟨Ŝ_j⟩ comes from outside (QuTiP), ⟨σ̂_j⟩ comes from spin_density() here.

GEOMETRY (src/hamiltonians.jl convention):
    linear site index  l = (col-1)*Ny + row ,  col = 1..Nx (x), row = 1..Ny (y)
    Nx = 7 columns, Ny = 2 rows. Left lead → col 1, right lead → col Nx.

SIGN CONVENTION WARNING:
    the package's update_H_e! implements   H = H0 - j_sd * (S · σ)
    i.e. a MINUS. If your QSL side defines the coupling as +J_s S·σ, pass
    j_sd = -J_s. I have not assumed which one you want — see apply_sd_field!.
=#

using Pkg
# Pkg.activate(joinpath(dirname((@__DIR__))))   # adjust to your setup

using TDNEGF
using DifferentialEquations
using LinearAlgebra
using StaticArrays

# ----------------------------------------------------------------------------
# Parameters (code units: hopping γ = 1, time unit ħ/γ)
# ----------------------------------------------------------------------------
Base.@kwdef struct NMConfig
    Nx::Int      = 7        # columns
    Ny::Int      = 2        # rows
    Nσ::Int      = 2
    N_orb::Int   = 1
    γ::Float64   = 1.0                   # device + lead hopping
    γso::ComplexF64 = 0.0 + 0.0im        # Rashba; 0 → no intrinsic spin source
    γ_lead::Float64 = 1.0
    μ::Float64   = 0.0                   # SAME in both leads (zero bias)
    Temp::Float64 = 0.03                 # in units of γ
    N_λ1::Int    = 49
    N_λ2::Int    = 20
    reltol::Float64 = 1e-6
    abstol::Float64 = 1e-8
end

# ----------------------------------------------------------------------------
# Device container: everything the driver needs to keep alive between steps
# ----------------------------------------------------------------------------
mutable struct NMDevice
    cfg::NMConfig
    p_model                       # ModelParamsTDNEGF
    p_blocks                      # ExperimentalBlockRHSParams
    integrator                    # DifferentialEquations integrator, held open
    scratch                       # 1-slice ObservablesTDNEGF used as a readout buffer
    site_ranges::Vector{UnitRange{Int}}
end

# index helpers
lin_idx(dev::NMDevice, col::Int, row::Int) = (col - 1) * dev.cfg.Ny + row
colrow(dev::NMDevice, l::Int) = ((l - 1) ÷ dev.cfg.Ny + 1, (l - 1) % dev.cfg.Ny + 1)

# ----------------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------------
"""
    nm_setup(cfg = NMConfig()) -> NMDevice

Build the 2×7 NM ladder with two equal-μ leads and open an integrator at t = 0
with the partitioned initial condition (empty device, coupling on at t = 0).
"""
function nm_setup(cfg::NMConfig = NMConfig())
    β = 1.0 / cfg.Temp
    Rλ, zλ = load_poles_square(cfg.N_λ1, cfg.N_λ2)

    p_model = ModelParamsTDNEGF(Nx=cfg.Nx, Ny=cfg.Ny, Nσ=cfg.Nσ, N_orb=cfg.N_orb,
                                Nα=2, N_λ1=cfg.N_λ1, N_λ2=cfg.N_λ2)

    # bare device Hamiltonian → H0_ab is the REFERENCE that update_H_e! rebuilds from
    H_ab = build_H_ab(; Nx=cfg.Nx, Ny=cfg.Ny, Nσ=cfg.Nσ, N_orb=cfg.N_orb,γ=cfg.γ, γso=cfg.γso)
    p_model.H0_ab .= H_ab
    p_model.H_ab  .= H_ab

    # both leads share μ, so one set of coefficients suffices
    Σᴸ = build_Σᴸ_nλ(Rλ, zλ, cfg.Ny, cfg.Nσ, cfg.N_orb, cfg.N_λ1, cfg.N_λ2;β=β, γ=cfg.γ_lead, μ=cfg.μ)
    Σᴳ = build_Σᴳ_nλ(Rλ, zλ, cfg.Ny, cfg.Nσ, cfg.N_orb, cfg.N_λ1, cfg.N_λ2;β=β, γ=cfg.γ_lead, μ=cfg.μ)
    χ  = build_χ_nλ(zλ,      cfg.Ny, cfg.Nσ, cfg.N_orb, cfg.N_λ1, cfg.N_λ2;β=β, γ=cfg.γ_lead, μ=cfg.μ)

    ξ_L = build_ξ_an(cfg.Nx, cfg.Ny, cfg.Nσ, cfg.N_orb; xcol=1,      y_coup=1:cfg.Ny)
    ξ_R = build_ξ_an(cfg.Nx, cfg.Ny, cfg.Nσ, cfg.N_orb; xcol=cfg.Nx, y_coup=1:cfg.Ny)

    left  = SelfEnergyBlock(:left,  p_model.Nc, cfg.N_λ1, cfg.N_λ2, Σᴸ, Σᴳ, χ, ξ_L)
    right = SelfEnergyBlock(:right, p_model.Nc, cfg.N_λ1, cfg.N_λ2, Σᴸ, Σᴳ, χ, ξ_R)
    blocks   = [left, right]
    Δ_blocks = ComplexF64[0.0, 0.0]

    # NOTE: this constructor stores p_model.H_ab BY REFERENCE (verified in
    # src/eom_tdnegf.jl: the field is assigned, not copied, and the RHS reads
    # p.H_ab on every call). So mutating p_model.H_ab in place is enough to
    # change the Hamiltonian seen by the integrator — no rebuild required.
    p_blocks = ExperimentalBlockRHSParams(p_model.H_ab, blocks, Δ_blocks, p_model)

    u0 = zeros(ComplexF64, p_blocks.dims_ρ_ab[1]^2 + p_blocks.aux_layout.total_size)

    # tspan end is a formality: we never call solve!, we call step! by hand.
    prob = ODEProblem(eom_tdnegf_blocks!, u0, (0.0, Inf), p_blocks)
    integ = init(prob, Vern7(); adaptive=true, save_everystep=false,
                 save_start=false, dense=false,
                 reltol=cfg.reltol, abstol=cfg.abstol)

    scratch = ObservablesTDNEGF(p_model; N_tmax=1, N_leads=length(blocks))
    scratch.t = [0.0]
    scratch.idx = 1

    site_ranges = [get_sub(i, p_model.N_loc) for i in 1:p_model.N_sites]

    return NMDevice(cfg, p_model, p_blocks, integ, scratch, site_ranges)
end

# ----------------------------------------------------------------------------
# Readout at the CURRENT time — this is the function you asked for
# ----------------------------------------------------------------------------
"""
    spin_density(dev) -> Matrix{Float64}  (N_sites × 3)

⟨σ̂_j⟩(t) = Tr_spin[ρ_jj σ̂] for every site j, at the integrator's current time.
Columns are (x, y, z). Row index j is the linear site index
`l = (col-1)*Ny + row`; use `colrow(dev, l)` to map back.

Units: this is Tr[ρ_loc σ], so a fully polarised site gives 1, not ħ/2.
"""
function spin_density(dev::NMDevice)
    ptr = pointer_blocks(dev.integrator.u, dev.p_blocks.dims_ρ_ab,
                         dev.p_blocks.aux_layout)
    dev.scratch.idx = 1
    obs_σ_i!(ptr, dev.p_model, dev.scratch)
    return copy(dev.scratch.σx_i[:, :, 1])      # (N_sites, 3)
end

"""
    spin_density_svec(dev) -> Vector{SVector{3,Float64}}

Same as `spin_density`, one SVector per site — the shape `update_H_e!` and most
spin-dynamics code wants.
"""
function spin_density_svec(dev::NMDevice)
    σ = spin_density(dev)
    return [SVector{3,Float64}(σ[l,1], σ[l,2], σ[l,3]) for l in 1:size(σ,1)]
end

"""
    spin_density_grid(dev) -> Matrix{SVector{3,Float64}}  (Nx × Ny)

⟨σ̂⟩ laid out on the (column, row) grid, matching the `S[a,b]` indexing that
`update_H_e!` expects.
"""
function spin_density_grid(dev::NMDevice)
    σ = spin_density(dev)
    G = Matrix{SVector{3,Float64}}(undef, dev.cfg.Nx, dev.cfg.Ny)
    for col in 1:dev.cfg.Nx, row in 1:dev.cfg.Ny
        l = lin_idx(dev, col, row)
        G[col, row] = SVector{3,Float64}(σ[l,1], σ[l,2], σ[l,3])
    end
    return G
end

"""
    charge_density(dev) -> Vector{Float64}  (N_sites,)

⟨n̂_j⟩ at the current time. SPIN-SUMMED (saturates near 2/site, not 1).
"""
function charge_density(dev::NMDevice)
    ptr = pointer_blocks(dev.integrator.u, dev.p_blocks.dims_ρ_ab,
                         dev.p_blocks.aux_layout)
    dev.scratch.idx = 1
    obs_n_i!(ptr, dev.p_model, dev.scratch)
    return copy(dev.scratch.n_i[:, 1])
end

"""
    lead_currents(dev) -> (Ic, Is)

`Ic::Vector{Float64}` charge current per lead, block order [left, right].
`Is::Matrix{Float64}` (N_leads × 3) spin current per lead, components (x,y,z).
"""
function lead_currents(dev::NMDevice)
    ptr = pointer_blocks(dev.integrator.u, dev.p_blocks.dims_ρ_ab,
                         dev.p_blocks.aux_layout)
    dev.scratch.idx = 1
    obs_Ixα!(ptr, dev.p_blocks, dev.scratch)
    return copy(dev.scratch.Iα[:, 1]), copy(dev.scratch.Iαx[:, :, 1])
end

current_time(dev::NMDevice) = dev.integrator.t

# ----------------------------------------------------------------------------
# Hamiltonian update — two ways in
# ----------------------------------------------------------------------------
"""
    set_hamiltonian!(dev, H)

Overwrite the device Hamiltonian in place with `H` (Ns × Ns, Ns = Nx*Ny*Nσ*N_orb).
Takes effect on the next `step!`. Does NOT touch H0_ab.
"""
function set_hamiltonian!(dev::NMDevice, H::AbstractMatrix)
    size(H) == size(dev.p_model.H_ab) ||
        throw(DimensionMismatch("H must be $(size(dev.p_model.H_ab))"))
    dev.p_model.H_ab .= H
    return dev
end

"""
    apply_sd_field!(dev, S, j_sd)

Rebuild H from the bare H0 plus the local exchange field:

        H = H0 - j_sd * Σ_j (S_j · σ̂_j)

`S` is an `Nx × Ny` matrix of `SVector{3,Float64}` — the ⟨Ŝ_j⟩ handed over by
the QSL side, indexed as `S[col, row]`.

SIGN: the package hard-codes the minus sign above. For the figure's
`+J_s ⟨Ŝ_j⟩·σ̂_j`, pass `j_sd = -J_s`. Confirm against your QSL convention.
"""
function apply_sd_field!(dev::NMDevice, S::Matrix{SVector{3,Float64}}, j_sd::Float64)
    size(S) == (dev.cfg.Nx, dev.cfg.Ny) ||
        throw(DimensionMismatch("S must be Nx×Ny = $((dev.cfg.Nx, dev.cfg.Ny))"))
    update_H_e!(dev.p_model, dev.site_ranges, S, j_sd)
    return dev
end

# ----------------------------------------------------------------------------
# The step
# ----------------------------------------------------------------------------
"""
    step!(dev, Δt)

Advance the electronic state from t to exactly t + Δt, holding H fixed at
whatever `set_hamiltonian!` / `apply_sd_field!` last put in `p_model.H_ab`.

This is the piecewise-constant-H scheme: H is frozen over each slice, so Δt
must be small compared with the timescale on which ⟨Ŝ_j⟩ moves (i.e. against
ω₀ and J_s). Halving Δt and checking nothing changes is the honest test.
"""
function step!(dev::NMDevice, Δt::Float64)
    Δt > 0 || throw(ArgumentError("Δt must be positive"))
    DifferentialEquations.step!(dev.integrator, Δt, true)   # stop exactly at t+Δt
    return dev.integrator.t
end

"""
    relax!(dev, T; Δt = 0.5)

Convenience: evolve for total time T with the current H, discarding output.
Use this once at the start to let the NM charge up from the empty initial
state before you switch on the coupling to the QSL.
"""
function relax!(dev::NMDevice, T::Float64; Δt::Float64 = 0.5)
    t_target = dev.integrator.t + T
    while dev.integrator.t < t_target - 1e-12
        step!(dev, min(Δt, t_target - dev.integrator.t))
    end
    return dev.integrator.t
end

# ----------------------------------------------------------------------------
# Minimal usage sketch — NOT executed on include()
# ----------------------------------------------------------------------------
function demo(; n_steps::Int = 20, Δt::Float64 = 0.5, J_s::Float64 = 0.1)
    dev = nm_setup()
    println("Ns = $(dev.p_model.Ns), N_sites = $(dev.p_model.N_sites)")

    relax!(dev, 20.0)                       # charge up the NM, decoupled
    println("after relax: t = $(current_time(dev)), n_1 = $(charge_density(dev)[1])")

    for k in 1:n_steps
        # --- this is where the QSL side would hand you ⟨Ŝ_j⟩ at the current t.
        #     Placeholder: uniform precessing field, just to exercise the path.
        t = current_time(dev)
        θ, ω0 = 0.3, 0.01
        Sv = SVector{3,Float64}(sin(θ)*cos(ω0*t), sin(θ)*sin(ω0*t), cos(θ))
        S  = fill(Sv, dev.cfg.Nx, dev.cfg.Ny)

        apply_sd_field!(dev, S, -J_s)       # H = H0 + J_s S·σ   (note the sign)
        step!(dev, Δt)

        σ = spin_density(dev)               # ← ⟨σ̂_j⟩(t) for ALL sites
        Ic, Is = lead_currents(dev)
        println("t = $(round(current_time(dev), digits=2))  ",
                "⟨σ⟩_site1 = $(round.(σ[1,:], digits=6))  ",
                "I_L = $(round(Ic[1], digits=6))")
    end
    return dev
end

# demo()   # uncomment to run