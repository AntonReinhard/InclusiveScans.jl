using Makie
using CairoMakie
using BenchmarkTools
using LaTeXStrings

include("utils.jl")

jsonfile = "benchmark_results.json"

plotpath = "plots"
if !isdir(plotpath)
    mkdir(plotpath)
end

yticks_v = [1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13]
yticks_str = [
    L"1ns",
    L"10ns",
    L"100ns",
    L"1μs",
    L"10μs",
    L"100μs",
    L"1ms",
    L"10ms",
    L"100ms",
    L"1s",
    L"10s",
    L"100s",
    L"1ks",
    L"10ks",
]


SIZES = [Int64(2^i) for i = 10:2:10]
#TYPES = (Float16, Float32, Float64, Int32, Int64, ComplexF32, ComplexF64)
TYPES = (Float64, Int64, ComplexF64)

_str(n::Int64) = L"2^{%$(round(Int, log2(n)))}"

# == CPU vs CUDA ==
result = BenchmarkTools.load(jsonfile)[1]

for T in TYPES
    extract_data =
        bench_data -> begin
            return mean.(getfield.(getindex.(Ref(copy(bench_data)), string.(SIZES)), :times))
        end

    cpu_data = extract_data(result["CPU"][T])
    cuda_data = extract_data(result["CUDA"][T])
    cuda_lib_data = extract_data(result["CUDA_LIB"][T])

    f = Figure()
    ax = Axis(
        f[1, 1];
        title = "Inclusive Scan of $T Arrays",
        xlabel = "input size (#)",
        ylabel = "time (s)",
        limits = (nothing, _find_y_lims([cpu_data, cuda_data, cuda_lib_data])),
        yminorgridvisible = true,
        yminorticksvisible = true,
        yminorticks = IntervalsBetween(10),
        xscale = log10,
        yscale = log10,
        xticks = (SIZES, _str.(SIZES)),
        yticks = (yticks_v, yticks_str),
    )

    sc_cpu = scatter!(ax, SIZES, cpu_data)
    sc_cuda_lib = scatter!(ax, SIZES, cuda_lib_data)
    sc_cuda = scatter!(ax, SIZES, cuda_data)

    # Legend
    labels = ["Julia accumulate!", "CUDA.jl accumulate!", "InclusiveScans on CUDA"]
    elements = [sc_cpu, sc_cuda_lib, sc_cuda]

    Legend(f[2, 1], elements, labels; orientation = :horizontal)

    save(joinpath(plotpath, "inclusive_scan_time_taken_$(T).pdf"), f)



    extract_data =
        bench_data -> begin
            return ./(
                mean.(getfield.(getindex.(Ref(copy(bench_data)), string.(SIZES)), :times)),
                SIZES,
            )
        end

    cpu_data = extract_data(result["CPU"][T])
    cuda_data = extract_data(result["CUDA"][T])
    cuda_lib_data = extract_data(result["CUDA_LIB"][T])

    f = Figure()
    ax = Axis(
        f[1, 1];
        title = "Inclusive Scan of $T Arrays",
        xlabel = "input size (#)",
        ylabel = "time per element (s)",
        limits = (nothing, _find_y_lims([cpu_data, cuda_data, cuda_lib_data])),
        yminorgridvisible = true,
        yminorticksvisible = true,
        yminorticks = IntervalsBetween(10),
        xscale = log10,
        yscale = log10,
        xticks = (SIZES, _str.(SIZES)),
        yticks = (yticks_v, yticks_str),
    )

    sc_cpu = scatter!(ax, SIZES, cpu_data)
    sc_cuda_lib = scatter!(ax, SIZES, cuda_lib_data)
    sc_cuda = scatter!(ax, SIZES, cuda_data)

    # Legend
    labels = ["Julia accumulate!", "CUDA.jl accumulate!", "InclusiveScans on CUDA"]
    elements = [sc_cpu, sc_cuda_lib, sc_cuda]

    Legend(f[2, 1], elements, labels; orientation = :horizontal)

    save(joinpath(plotpath, "inclusive_scan_time_per_element_$(T).pdf"), f)
end
