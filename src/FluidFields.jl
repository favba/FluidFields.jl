module FluidFields

using InplaceRealFFT, LinearAlgebra, FFTW
using FluidTensors
import FluidTensors

export Space, ScalarField, VectorField, SymTenField, SymTrTenField, isrealspace, setreal!, setfourier!, real!, fourier!, AbstractField, AbstractTensorField


const Float3264 = InplaceRealFFT.Float3264

include("space.jl")
include("scalar.jl")
include("vector.jl")
include("tensor.jl")

const AbstractField = Union{<:ScalarField,<:VectorField,<:SymTrTenField}

end # module
