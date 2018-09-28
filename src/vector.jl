struct VectorField{T,N,N2,L} <: AbstractVecArray{Complex{T},N}
    rr::VecArray{T, N, Array{T,N}, Array{T,N}, Array{T,N}}
    c::VecArray{Complex{T}, N, ScalarField{T,N,N2,L}, ScalarField{T,N,N2,L}, ScalarField{T,N,N2,L}}
    r::SubArray{Vec{T},N,VecArray{T,N,Array{T,N},Array{T,N},Array{T,N}},Tuple{Base.OneTo{Int},Vararg{Base.Slice{Base.OneTo{Int}},N2}},L} 
    function VectorField{T,N,N2,L}(x::ScalarField{T,N,N2,L},y::ScalarField{T,N,N2,L},z::ScalarField{T,N,N2,L}) where {T,N,N2,L}
        c = VecArray(x,y,z)
        rr = VecArray(InplaceRealFFT.data(x),InplaceRealFFT.data(y),InplaceRealFFT.data(z))
        r = view(rr, Base.OneTo(size(x.field.r, 1)), ntuple(i->Colon(), Val(N2))...)
        return new{T,N,N2,L}(rr,c,r)
    end
end

@inline FluidTensors.xvec(v::VectorField) =
    FluidTensors.xvec(v.c)
@inline FluidTensors.yvec(v::VectorField) =
    FluidTensors.yvec(v.c)
@inline FluidTensors.zvec(v::VectorField) =
    FluidTensors.zvec(v.c)

VectorField(x::ScalarField{T,N,N2,L},y::ScalarField{T,N,N2,L},z::ScalarField{T,N,N2,L}) where {T,N,N2,L} = VectorField{T,N,N2,L}(x,y,z)

VectorField{T}(dims::Vararg{Int}) where T = VectorField(ScalarField{T}(dims...),ScalarField{T}(dims...),ScalarField{T}(dims...))
VectorField(dims::Vararg{Int}) = VectorField(ScalarField(dims...),ScalarField(dims...),ScalarField(dims...))

Base.similar(a::VectorField) = VectorField(similar(a.c.x),similar(a.c.y),similar(a.c.z))

function InplaceRealFFT.rfft!(v::VectorField)
    rfft!(v.c.x)
    rfft!(v.c.y)
    rfft!(v.c.z)
    return v
end

function InplaceRealFFT.brfft!(v::VectorField)
    brfft!(v.c.x)
    brfft!(v.c.y)
    brfft!(v.c.z)
    return v.r
end

function InplaceRealFFT.irfft!(v::VectorField)
    irfft!(v.c.x)
    irfft!(v.c.y)
    irfft!(v.c.z)
    return v.r
end

isrealspace(a::VectorField) =
    isrealspace(a.c.x)

Base.show(io::IO,m::MIME"text/plain",a::VectorField) =
    isrealspace(a) ? show(io,m,a.r) : show(io,m,a.c)