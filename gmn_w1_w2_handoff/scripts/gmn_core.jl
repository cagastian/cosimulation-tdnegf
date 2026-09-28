# =============================================================================
# gmn_core.jl  —  computational core. Safe to `include` on every worker:
#                 nothing here runs on load.
# =============================================================================

using LinearAlgebra
using JuMP
using Mosek
using MosekTools
using DelimitedFiles
using Printf

# ============================================================== configuration

const IN_DIR  = get(ENV, "GMN_IN_DIR",  "GMN_results")
const OUT_DIR = get(ENV, "GMN_OUT_DIR", "GMN_calculation_res")

"""
Solver options. `nthreads` is Mosek's internal pool — see the note in
gmn_run.jl about budgeting it against the number of worker processes.

`order_method`: 3 = TRY_GRAPHPAR, 4 = FORCE_GRAPHPAR (per Mosek's
MSK_ORDER_METHOD enum). Set to `nothing` to leave it at the default; the
setter is wrapped in a try/catch in case your Mosek version rejects it.
"""
Base.@kwdef struct SolveOpts
    nthreads::Int                  = 16
    real_mode::Union{Symbol,Bool}  = :auto
    show_log::Bool                 = false
    order_method::Union{Int,Nothing} = 3
    tol::Float64                   = 1e-8
end

# ============================================================== bipartitions

function bipartitions(n::Int)
    parts = Vector{Vector{Int}}()
    for m in 1:(2^(n-1)-1)
        push!(parts, reverse(digits(m, base=2, pad=n)))
    end
    return parts
end

# ================================================= partial transpose as lookup
#
# Q_PT[r,c] = Q[rmap[r,c], cmap[r,c]]: the subsystem indices inside the
# partition are swapped between row and column. Involution, so the same map
# applies in both directions. Pure lookup, no symbolic sparse matvec.

function PT_index_maps(Mvec::Vector{Int}, dims::Vector{Int})
    n, d, dim = length(dims), prod(dims), dims[1]
    # The base-`dim` digit expansion below is only valid for equal local dims.
    @assert all(==(dim), dims) "PT_index_maps assumes all local dimensions equal; got $dims"

    rmap = zeros(Int, d, d)
    cmap = zeros(Int, d, d)
    i = zeros(Int, n)
    j = zeros(Int, n)
    ii = zeros(Int, n)
    jj = zeros(Int, n)

    for c in 1:d
        digits!(j, c-1, base=dim)
        for r in 1:d
            digits!(i, r-1, base=dim)
            copyto!(ii, i); copyto!(jj, j)
            for s in 1:n
                if Mvec[s] == 1
                    ii[s], jj[s] = jj[s], ii[s]
                end
            end
            rmap[r, c] = 1 + sum(ii[s] * dim^(s-1) for s in 1:n)
            cmap[r, c] = 1 + sum(jj[s] * dim^(s-1) for s in 1:n)
        end
    end
    return rmap, cmap
end

"""Threaded over bipartitions. Cheap at d=64; matters if you push to 8 qubits."""
function build_all_PT_maps(dims::Vector{Int})
    parts = bipartitions(length(dims))
    maps  = Vector{Tuple{Matrix{Int},Matrix{Int}}}(undef, length(parts))
    Threads.@threads for k in eachindex(parts)
        maps[k] = PT_index_maps(parts[k], dims)
    end
    return maps
end

# ======================================================================== I/O

function load_rho(filename::AbstractString)
    rho_r = readdlm(joinpath(IN_DIR, filename * "_r.txt"))
    rho_i = readdlm(joinpath(IN_DIR, filename * "_i.txt"))
    @assert size(rho_r) == size(rho_i) "r/i shape mismatch: $(size(rho_r)) vs $(size(rho_i))"
    ρ = rho_r + 1im * rho_i
    return (ρ + ρ') / 2
end

is_effectively_real(ρ; atol = 1e-9) = maximum(abs, imag(ρ)) < atol

"""Substitute `{j}` in a filename template."""
expand_template(template::AbstractString, j) = replace(template, "{j}" => string(j))

# Linux-only; returns NaN elsewhere. VmHWM = peak resident set size.
function peak_rss_mb()
    Sys.islinux() || return NaN
    try
        for line in eachline("/proc/self/status")
            if startswith(line, "VmHWM:")
                return parse(Float64, split(line)[2]) / 1024
            end
        end
    catch
    end
    return NaN
end

# ================================================================ model reuse
#
# The model structure depends only on (dims, real_mode). Across time points
# ONLY the objective changes, so we build once and re-`@objective` per solve.

struct GMNModel
    model::Model
    W
    d::Int
    real_mode::Bool
    build_time::Float64
end

function build_gmn_model(dims::Vector{Int}, PT_maps; real_mode::Bool, opts::SolveOpts)
    d = prod(dims)

    t = @elapsed begin
        model = Model(Mosek.Optimizer)
        opts.show_log || set_silent(model)

        # Threading must be set before the first optimize!: for conic problems
        # Mosek reserves its pool at the first solve and ignores later changes.
        set_optimizer_attribute(model, "MSK_IPAR_NUM_THREADS", opts.nthreads)

        if opts.order_method !== nothing
            try
                set_optimizer_attribute(model, "MSK_IPAR_INTPNT_ORDER_METHOD",
                                        opts.order_method)
            catch err
                @warn "MSK_IPAR_INTPNT_ORDER_METHOD not accepted; using default" err
            end
        end

        if real_mode
            @variable(model, W[1:d, 1:d], Symmetric)
            cone, wrap = PSDCone(), Symmetric
        else
            @variable(model, W[1:d, 1:d], Hermitian)
            cone, wrap = HermitianPSDCone(), Hermitian
        end

        @constraint(model, tr(W) == 1)

        for (rmap, cmap) in PT_maps
            Qm    = @variable(model, [1:d, 1:d] in cone)
            Qm_PT = [Qm[rmap[r, c], cmap[r, c]] for r in 1:d, c in 1:d]
            @constraint(model, wrap(I(d) - Qm) in cone)
            @constraint(model, wrap(W - Qm_PT)  in cone)
        end
    end

    return GMNModel(model, W, d, real_mode, t)
end

"""Swap in a new objective and re-solve. Returns value, witness, status, timing."""
function solve_for!(gm::GMNModel, rho::AbstractMatrix; opts::SolveOpts)
    d = gm.d
    @assert size(rho) == (d, d) "ρ is $(size(rho)) but model has d=$d"
    @assert norm(rho - rho') < 1e-10

    if gm.real_mode
        R = real(rho)
        @objective(gm.model, Min, sum(R[r, c] * gm.W[r, c] for r in 1:d, c in 1:d))
    else
        # tr(ρW) = Σ_{r,c} ρ[r,c] W[c,r]. O(d²) symbolic terms instead of the
        # O(d³) you get from forming the product `rho * W` first.
        @objective(gm.model, Min,
                   real(sum(rho[r, c] * gm.W[c, r] for r in 1:d, c in 1:d)))
    end

    t = @elapsed optimize!(gm.model)

    status = termination_status(gm.model)
    status == MOI.OPTIMAL || @warn "solver did not reach OPTIMAL" status

    minval = objective_value(gm.model)
    Wopt   = minval < -opts.tol ? value.(gm.W) : zeros(d, d)
    return (minval = minval, W = Wopt, status = status, solve_time = t)
end

# ================================================================ chunk runner
#
# A "chunk" = one (dims, contiguous range of time points). The model is built
# once per chunk and reused across the time points inside it. This is the knob
# that trades model-reuse against parallel width — see gmn_run.jl.

struct Job
    dims::Vector{Int}
    name::String
    js::Vector{Int}
    template::String
end

function run_chunk(job::Job, opts::SolveOpts)
    dims, js = job.dims, job.js
    @info "chunk start" job.name js dims threads=opts.nthreads worker=myid()

    t_maps  = @elapsed PT_maps = build_all_PT_maps(dims)
    t_load  = @elapsed rhos = [load_rho(expand_template(job.template, j)) for j in js]

    # Decide real/complex ONCE for the whole chunk: the model structure is
    # fixed, so every ρ in the chunk must agree. Complex is always valid, so
    # fall back to it unless all of them are effectively real.
    rm = if opts.real_mode === :auto
        all(is_effectively_real, rhos)
    else
        opts.real_mode::Bool
    end
    if rm && !all(is_effectively_real, rhos)
        worst = maximum(ρ -> maximum(abs, imag(ρ)), rhos)
        error("real_mode forced on, but some ρ in this chunk has imaginary part $worst")
    end

    gm = build_gmn_model(dims, PT_maps; real_mode = rm, opts = opts)
    @info "model built" job.name build_min=round(gm.build_time/60, digits=3) real_mode=rm

    vals        = zeros(length(js))
    solve_times = zeros(length(js))

    for (k, j) in enumerate(js)
        res            = solve_for!(gm, rhos[k]; opts = opts)
        vals[k]        = -res.minval
        solve_times[k] = res.solve_time
        @info "solved" name=job.name j GMN=round(vals[k], sigdigits=6) min=round(res.solve_time/60, digits=2)
    end

    # Per-chunk checkpoint: no write races between workers, and a crash keeps
    # whatever finished.
    mkpath(OUT_DIR)
    writedlm(joinpath(OUT_DIR,
             @sprintf("partial_%sj%d-%d.txt", job.name, first(js), last(js))),
             hcat(js, vals))

    return (name = job.name, js = js, vals = vals,
            build_time = gm.build_time, solve_times = solve_times,
            t_maps = t_maps, t_load = t_load,
            peak_rss_mb = peak_rss_mb(), real_mode = rm, worker = myid())
end

# ================================================================= diagnostics

"""
Build + solve a single instance with the Mosek log on, reporting build time,
solve time and peak RSS separately. Run this BEFORE choosing worker/thread
counts — these three numbers determine the whole configuration.
"""
function benchmark(dims::Vector{Int}, template::String; j = 1,
                   opts::SolveOpts = SolveOpts(show_log = true))
    PT_maps = build_all_PT_maps(dims)
    ρ       = load_rho(expand_template(template, j))
    rm      = opts.real_mode === :auto ? is_effectively_real(ρ) : opts.real_mode::Bool

    gm  = build_gmn_model(dims, PT_maps; real_mode = rm, opts = opts)
    res = solve_for!(gm, ρ; opts = opts)

    @printf("\n---- benchmark dims=%s  d=%d  bipartitions=%d  real=%s  threads=%d\n",
            string(dims), prod(dims), length(PT_maps), rm, opts.nthreads)
    @printf("  model build : %8.2f s\n", gm.build_time)
    @printf("  optimize!   : %8.2f s\n", res.solve_time)
    @printf("  build/solve : %8.3f   (high => chunk fewer, reuse more)\n",
            gm.build_time / max(res.solve_time, eps()))
    @printf("  peak RSS    : %8.1f MB  (÷ this into your RAM = max workers)\n",
            peak_rss_mb())
    return (build = gm.build_time, solve = res.solve_time, rss = peak_rss_mb())
end

"""Sweep Mosek's thread count on one instance. Rebuilds the model each time
because the pool is fixed at the first solve."""
function sweep_threads(dims::Vector{Int}, template::String,
                       threads_list = [4, 8, 16, 32, 64]; j = 1)
    PT_maps = build_all_PT_maps(dims)
    ρ       = load_rho(expand_template(template, j))
    rm      = is_effectively_real(ρ)

    println("\n---- thread sweep, dims=$dims, j=$j")
    base = NaN
    for nt in threads_list
        o   = SolveOpts(nthreads = nt, show_log = false)
        gm  = build_gmn_model(dims, PT_maps; real_mode = rm, opts = o)
        res = solve_for!(gm, ρ; opts = o)
        isnan(base) && (base = res.solve_time)
        @printf("  threads=%3d   solve=%8.2f s   speedup=%5.2fx   eff=%4.0f%%\n",
                nt, res.solve_time, base / res.solve_time,
                100 * (base / res.solve_time) / (nt / first(threads_list)))
    end
end
