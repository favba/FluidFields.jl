module FluidFields

using InplaceRealFFT, LinearAlgebra, FFTW
using FluidTensors
import FluidTensors

export ScalarField, VectorField, SymTenField, SymTrTenField, isrealspace, setreal!, setfourier!, real!, fourier!, AbstractField, AbstractTensorField


const Float3264 = InplaceRealFFT.Float3264

include("kvec.jl")
include("scalar.jl")
include("vector.jl")
include("tensor.jl")

const AbstractField = Union{<:ScalarField,<:VectorField,<:SymTrTenField}

function __init__()
    path = @__DIR__
    fftwplan = joinpath(path,"fftw_wisdom")
    FFTW.set_num_threads(Threads.nthreads())
    if isfile(fftwplan)
        try 
            FFTW.import_wisdom(fftwplan)
        catch
        end
    end
    atexit(()->(FFTW.export_wisdom(fftwplan)))
end

end # module
