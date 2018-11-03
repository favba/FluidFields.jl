module FluidFields

using InplaceRealFFT, LinearAlgebra, FFTW
using FluidTensors
import FluidTensors

export Space, ScalarField, VectorField, SymTrTenField, isrealspace, setreal!, setfourier!, real!, fourier!

const Float3264 = InplaceRealFFT.Float3264

include("space.jl")
include("scalar.jl")
include("vector.jl")
include("tensor.jl")

end # module
