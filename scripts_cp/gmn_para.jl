# =============================================================================
# gmn_run.jl  —  driver. Run with:  julia -t 4 gmn_run.jl
#
# Environment knobs (all optional):
#   GMN_WORKERS = 1     number of worker processes
#   GMN_THREADS = 16    Mosek threads PER worker  (WORKERS × THREADS ≈ cores)
#   GMN_CHUNKS  = 9     time-point chunks per partition (see note below)
#   GMN_IN_DIR  / GMN_OUT_DIR
#
# MEMORY, NOT CPU, IS THE CONSTRAINT HERE -- READ BEFORE RAISING GMN_WORKERS
# ---------------------------------------------------------------------------
# Confirmed by two separate incidents on this 1 TiB machine:
#   - a prior single job (9 solves reusing one model) peaked at ~742 GB RSS.
#   - trying 4 CONCURRENT single-solve jobs here OOM-killed one worker within
#     under a minute of it starting `optimize!` (RSS still climbing through
#     ~297 GB at kill time), and the other 3 kept ballooning past 947 GB
#     combined before being killed by hand -- swap (63 GB) filled completely,
#     and it nearly took a co-running run_coupled.py job down with it.
# So: a SINGLE solve for this 64-dim/31-bipartition problem can apparently
# approach or exceed half of 1 TiB on its own. GMN_WORKERS=1 (strictly
# sequential, one Mosek instance at a time) is the only configuration
# confirmed safe -- do not raise it without a real fix to the footprint.
#
# CHUNKS=9 (vs. 1) is still worth keeping even at WORKERS=1: 18 independent
# per-time-point jobs means a crash only loses that one point (checkpointed
# to partial_*.txt), instead of losing an entire 9-solve chunk.
# =============================================================================

using Distributed
using Printf

const NWORKERS = parse(Int, get(ENV, "GMN_WORKERS", "1"))
const NTHREADS = parse(Int, get(ENV, "GMN_THREADS", "16"))
const NCHUNKS  = parse(Int, get(ENV, "GMN_CHUNKS",  "9"))

if nprocs() < NWORKERS + 1
    addprocs(NWORKERS + 1 - nprocs();
             exeflags = ["--project=$(dirname(Base.active_project()))",
                         "-t", string(max(1, Threads.nthreads()))])
end

@everywhere include(joinpath(@__DIR__, "gmn_core.jl"))

# Each worker runs Mosek with its own pool; a second BLAS pool per worker would
#
# oversubscribe the machine badly.
@everywhere LinearAlgebra.BLAS.set_num_threads(1)

# ===================================================================== problem
#
# Two runs, same 6-site hexagon (dims=[2,2,2,2,2,2]) exported by
# run_coupled.py's export_rho_for_gmn: FM_jK0.05_env5-40_kondo15_julia and
# AFM_jK0.05_env5-40_kondo15_julia, 9 GMN_results snapshots each
# ({prefix}_{j}_1_theta0.393_[ri].txt, j=1..9).

const ALL_RUNS = [
    (name = "FM_",  dims = [2, 2, 2, 2, 2, 2],
     template = "FM_jK0.05_env5-40_kondo15_julia_{j}_1_theta0.393"),
    (name = "AFM_", dims = [2, 2, 2, 2, 2, 2],
     template = "AFM_jK0.05_env5-40_kondo15_julia_{j}_1_theta0.393"),
]
# GMN_ONLY="FM_" (or "AFM_", or "FM_,AFM_") restricts which of the above run
# in this invocation -- e.g. AFM's rho isn't exported until its run_coupled.py
# job finishes, so a first invocation can cover FM_ alone.
const ONLY = split(get(ENV, "GMN_ONLY", ""), ",", keepempty=false)
const RUNS = isempty(ONLY) ? ALL_RUNS : filter(r -> r.name in ONLY, ALL_RUNS)
const JS = collect(1:9)

chunk(v, n) = [v[i:min(i + cld(length(v), n) - 1, end)]
               for i in 1:cld(length(v), n):length(v)]

function build_jobs()
    jobs = Job[]
    for r in RUNS, js in chunk(JS, NCHUNKS)
        push!(jobs, Job(r.dims, r.name, js, r.template))
    end
    # Heaviest first: pmap hands out work in order, so this keeps the long
    # 6-qubit jobs from starting last and leaving everyone else idle.
    sort!(jobs, by = j -> -prod(j.dims))
    return jobs
end

# ======================================================================== main

function run_all()
    mkpath(OUT_DIR)
    opts = SolveOpts(nthreads = NTHREADS)
    jobs = build_jobs()

    @printf("workers=%d  mosek_threads_each=%d  chunks=%d  jobs=%d\n",
            NWORKERS, NTHREADS, NCHUNKS, length(jobs))

    t_wall  = @elapsed results = pmap(j -> run_chunk(j, opts), jobs)

    # Reassemble per run, in time order.
    all_vals = Dict{String,Vector{Float64}}()
    for r in RUNS
        name = r.name
        vals = fill(NaN, length(JS))
        for res in results
            res.name == name || continue
            for (k, j) in enumerate(res.js)
                vals[findfirst(==(j), JS)] = res.vals[k]
            end
        end
        all_vals[name] = vals
        writedlm(joinpath(OUT_DIR, name * "GMN.txt"), vals)
    end

    # ---- timing report
    println("\n---- timing")
    for r in results
        @printf("  %-18s js=%-8s worker=%d  build=%6.1fs  solve=%7.1fs  rss=%7.1fMB\n",
                r.name, string(first(r.js)) * "-" * string(last(r.js)), r.worker,
                r.build_time, sum(r.solve_times), r.peak_rss_mb)
    end
    cpu = sum(r -> r.build_time + sum(r.solve_times), results)
    @printf("  wall = %.1f s   cpu-serial-equivalent = %.1f s   speedup = %.2fx\n",
            t_wall, cpu, cpu / t_wall)

    return all_vals
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_all()
end
