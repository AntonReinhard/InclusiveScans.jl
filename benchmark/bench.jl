using BenchmarkTools
using InclusiveScans
using CUDA
using Random

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 10.0

RNG = Xoshiro(137)  # Fixed seed
SIZES = [Int64(2^i) for i = 0:2:28]
TYPES = (Float16, Float32, Float64, Int32, Int64, ComplexF32, ComplexF64)

SUITE = BenchmarkGroup()

SUITE["CPU"] = BenchmarkGroup()
SUITE["CUDA"] = BenchmarkGroup()
SUITE["CUDA_LIB"] = BenchmarkGroup()

for T in TYPES
    SUITE["CPU"][T] = BenchmarkGroup()
    SUITE["CUDA"][T] = BenchmarkGroup()
    SUITE["CUDA_LIB"][T] = BenchmarkGroup()

    for N in SIZES
        # Generate random input
        h_in = rand(RNG, T, N)
        h_out = similar(h_in)

        d_in = CuArray(h_in)
        d_out = similar(d_in)

        # Run inclusive scan on GPU
        SUITE["CUDA"][T][N] =
            @benchmarkable InclusiveScans.largeArrayScanInclusive!(out, in, Int32(n)) setup =
                (in = $d_in; out = $d_out; n = $N)

        SUITE["CUDA_LIB"][T][N] =
            @benchmarkable CUDA.accumulate!(+, out, in) setup = (in = $d_in; out = $d_out)

        SUITE["CPU"][T][N] =
            @benchmarkable Base.accumulate!(+, out, in) setup = (in = $h_in; out = $h_out)
    end
end

tune!(SUITE; verbose = true)
result = run(SUITE; verbose = true)
BenchmarkTools.save("benchmark_results.json", result)
