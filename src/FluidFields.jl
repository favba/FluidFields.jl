module FluidFields

using InplaceRealFFT, LinearAlgebra, FFTW
using FluidTensors
import FluidTensors

export ScalarField, VectorField, SymTrTenField, isrealspace

const Float3264 = InplaceRealFFT.Float3264

include("scalar.jl")
include("vector.jl")
include("tensor.jl")

end # module
