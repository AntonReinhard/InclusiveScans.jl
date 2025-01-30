"""
    _find_y_lims(data)
Return the limits which encloses the data, i.e. the supremum and infimum of the data in powers of 10.
This is useful for fining limit of logarithmic data.
"""
function _find_y_lims(data::BenchmarkTools.BenchmarkGroup)
    t_min = time(minimum(data))
    data_min = minimum(values(t_min))
    ymin = 10^floor(log10(data_min))

    t_max = time(maximum(data))
    data_max = maximum(values(t_max))
    ymax = 10^ceil(log10(data_max))

    return (ymin, ymax)
end

function _find_y_lims(data::AbstractVector{<:Number})
    y_min = minimum(data)
    y_min = 10^floor(log10(y_min))

    y_max = maximum(data)
    y_max = 10^ceil(log10(y_max))

    return (y_min, y_max)
end

function _find_y_lims(data::AbstractVector{T}) where {T}
    local y_min = Inf64
    local y_max = -Inf64
    for t in data
        (y_min_n, y_max_n) = _find_y_lims(t)
        y_min = min(y_min_n, y_min)
        y_max = max(y_max_n, y_max)
    end

    return (y_min, y_max)
end
