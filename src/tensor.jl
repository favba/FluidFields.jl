struct SymTrTenField{T,N,N2,L} <: AbstractSymTrTenArray{T,N}
    rr::SymTrTenArray{T,N,Array{T,N},Array{T,N},Array{T,N},Array{T,N},Array{T,N}}
    c::SymTrTenArray{Complex{T},N,ScalarField{T,N,N2,L},ScalarField{T,N,N2,L},ScalarField{T,N,N2,L},ScalarField{T,N,N2,L},ScalarField{T,N,N2,L}}
    r::SubArray{SymTen{T},N,SymTrTenArray{T,N,Array{T,N},Array{T,N},Array{T,N},Array{T,N},Array{T,N}},Tuple{Base.OneTo{Int},Vararg{Base.Slice{Base.OneTo{Int}},N2}},L} 

    function SymTrTenField{T,N,N2,L}(xx::ScalarField{T,N,N2,L},xy::ScalarField{T,N,N2,L},xz::ScalarField{T,N,N2,L},yy::ScalarField{T,N,N2,L},yz::ScalarField{T,N,N2,L}) where {T,N,N2,L}
        c = SymTrTenArray(xx,xy,xz,yy,yz)
        rr = SymTrTenArray(InplaceRealFFT.data(xx),InplaceRealFFT.data(xy),InplaceRealFFT.data(xz),InplaceRealFFT.data(yy),InplaceRealFFT.data(yz))
        r = view(rr, Base.OneTo(size(xx.field.r, 1)), ntuple(i->Colon(), Val(N2))...)
        return new{T,N,N2,L}(rr,c,r)
    end

end

@inline function Base.getproperty(a::S,s::Symbol) where {S<:SymTrTenField}
    if (s === :kx || s === :ky || s === :kz || s === :k)
        return getfield(getfield(getfield(getfield(a,:c),:xx),:space),s)
    else
        return getfield(a,s)
    end
end

@inline FluidTensors.xxvec(v::SymTrTenField) =
    FluidTensors.xxvec(v.c)
@inline FluidTensors.xyvec(v::SymTrTenField) =
    FluidTensors.xyvec(v.c)
@inline FluidTensors.xzvec(v::SymTrTenField) =
    FluidTensors.xzvec(v.c)
@inline FluidTensors.yyvec(v::SymTrTenField) =
    FluidTensors.yyvec(v.c)
@inline FluidTensors.yzvec(v::SymTrTenField) =
    FluidTensors.yzvec(v.c)

SymTrTenField(xx::ScalarField{T,N,N2,L},xy::ScalarField{T,N,N2,L},xz::ScalarField{T,N,N2,L},yy::ScalarField{T,N,N2,L},yz::ScalarField{T,N,N2,L}) where {T,N,N2,L} = SymTrTenField{T,N,N2,L}(xx,xy,xz,yy,yz)

SymTrTenField{T}(dims::NTuple{3,Int},l::NTuple{3,Real}) where T = SymTrTenField(ScalarField{T}(dims,l),ScalarField{T}(dims,l),ScalarField{T}(dims,l),ScalarField{T}(dims,l),ScalarField{T}(dims,l))
SymTrTenField(dims::NTuple{3,Int},l::NTuple{3,Real}) = SymTrTenField(ScalarField(dims,l),ScalarField(dims,l),ScalarField(dims,l),ScalarField(dims,l),ScalarField(dims,l))

Base.similar(a::SymTrTenField) = SymTrTenField(similar(a.c.xx),similar(a.c.xy),similar(a.c.xz),similar(a.c.yy),similar(a.c.yz))

function InplaceRealFFT.rfft!(v::SymTrTenField)
    rfft!(v.c.xx)
    rfft!(v.c.xy)
    rfft!(v.c.xz)
    rfft!(v.c.yy)
    rfft!(v.c.yz)
    return v
end

function InplaceRealFFT.brfft!(v::SymTrTenField)
    brfft!(v.c.xx)
    brfft!(v.c.xy)
    brfft!(v.c.xz)
    brfft!(v.c.yy)
    brfft!(v.c.yz)
    return v.r
end

function InplaceRealFFT.irfft!(v::SymTrTenField)
    irfft!(v.c.xx)
    irfft!(v.c.xy)
    irfft!(v.c.xz)
    irfft!(v.c.yy)
    irfft!(v.c.yz)
    return v.r
end

isrealspace(a::SymTrTenField) =
    isrealspace(a.c.xx) && isrealspace(a.c.xy) && isrealspace(a.c.xz) && isrealspace(a.c.yy) && isrealspace(a.c.yz)

Base.show(io::IO,m::MIME"text/plain",a::SymTrTenField) =
    isrealspace(a) ? show(io,m,a.r) : show(io,m,a.c)

setreal!(a::SymTrTenField) = (setreal!(a.c.xx); setreal!(a.c.xy); setreal!(a.c.xz); setreal!(a.c.yy); setreal!(a.c.yz))

setfourier!(a::SymTrTenField) = (setfourier!(a.c.xx); setfourier!(a.c.xy); setfourier!(a.c.xz); setfourier!(a.c.yy); setfourier!(a.c.yz))

fourier!(a::SymTrTenField) = (fourier!(a.c.xx); fourier!(a.c.xy); fourier!(a.c.xz); fourier!(a.c.yy); fourier!(a.c.yz); a)

real!(a::SymTrTenField) = (real!(a.c.xx); real!(a.c.xy); real!(a.c.xz); real!(a.c.yy); real!(a.c.yz); a.r)