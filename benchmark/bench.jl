using BenchmarkTools
using InclusiveScans
using CUDA
using Random

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 10.0

RNG = Xoshiro(137)  # Fixed seed
SIZES = [Int64(2^i) for i = 0:2:30]
TYPES = (Float16, Float32, Float64, Int32, Int64, ComplexF32)

SUITE = BenchmarkGroup()
result = BenchmarkGroup()

SUITE["CPU"] = BenchmarkGroup()
SUITE["CUDA"] = BenchmarkGroup()
SUITE["CUDA_LIB"] = BenchmarkGroup()

for T in TYPES
    SUITE["CPU"][T] = BenchmarkGroup()
    SUITE["CUDA"][T] = BenchmarkGroup()
    SUITE["CUDA_LIB"][T] = BenchmarkGroup()

    for N in SIZES
        @info "Type $T Size $N"

        # Generate random input
        h_in = rand(RNG, T, N)
        h_out = similar(h_in)

        d_in = CuArray(h_in)
        d_out = similar(d_in)

        # Run inclusive scan on GPU
        cuda_b = @benchmarkable (CUDA.@sync InclusiveScans.largeArrayScanInclusive!(
            out,
            in,
            Int32(n),
        )) setup = (in = $d_in; out = $d_out; n = $N)
        tune!(cuda_b)
        SUITE["CUDA"][T][N] = run(cuda_b)

        cuda_lib_b = @benchmarkable (CUDA.@sync CUDA.accumulate!(+, out, in)) setup =
            (in = $d_in; out = $d_out)
        tune!(cuda_lib_b)
        SUITE["CUDA_LIB"][T][N] = run(cuda_lib_b)

        cpu_b =
            @benchmarkable Base.accumulate!(+, out, in) setup = (in = $h_in; out = $h_out)
        tune!(cpu_b)
        SUITE["CPU"][T][N] = run(cpu_b)
    end
end

BenchmarkTools.save("benchmark_results.json", SUITE)
