struct SymTrTenField{T,N,N2,L} <: AbstractSymTrTenArray{Complex{T},N}
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
        return getproperty(getfield(getfield(a,:c),:xx),s)
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


################################ SymTenField ########################################

struct SymTenField{T,N,N2,L} <: AbstractSymTenArray{Complex{T},N}
    rr::SymTenArray{T,N,Array{T,N},Array{T,N},Array{T,N},Array{T,N},Array{T,N},Array{T,N}}
    c::SymTenArray{Complex{T},N,ScalarField{T,N,N2,L},ScalarField{T,N,N2,L},ScalarField{T,N,N2,L},ScalarField{T,N,N2,L},ScalarField{T,N,N2,L},ScalarField{T,N,N2,L}}
    r::SubArray{SymTen{T},N,SymTenArray{T,N,Array{T,N},Array{T,N},Array{T,N},Array{T,N},Array{T,N},Array{T,N}},Tuple{Base.OneTo{Int},Vararg{Base.Slice{Base.OneTo{Int}},N2}},L} 

    function SymTenField{T,N,N2,L}(xx::ScalarField{T,N,N2,L},xy::ScalarField{T,N,N2,L},xz::ScalarField{T,N,N2,L},yy::ScalarField{T,N,N2,L},yz::ScalarField{T,N,N2,L},zz::ScalarField{T,N,N2,L}) where {T,N,N2,L}
        c = SymTenArray(xx,xy,xz,yy,yz,zz)
        rr = SymTenArray(InplaceRealFFT.data(xx),InplaceRealFFT.data(xy),InplaceRealFFT.data(xz),InplaceRealFFT.data(yy),InplaceRealFFT.data(yz),InplaceRealFFT.data(zz))
        r = view(rr, Base.OneTo(size(xx.field.r, 1)), ntuple(i->Colon(), Val{N2}())...)
        return new{T,N,N2,L}(rr,c,r)
    end

end

@inline function Base.getproperty(a::S,s::Symbol) where {S<:SymTenField}
    if (s === :kx || s === :ky || s === :kz || s === :k)
        return getproperty(getfield(getfield(a,:c),:xx),s)
    else
        return getfield(a,s)
    end
end

@inline FluidTensors.xxvec(v::SymTenField) =
    FluidTensors.xxvec(v.c)
@inline FluidTensors.xyvec(v::SymTenField) =
    FluidTensors.xyvec(v.c)
@inline FluidTensors.xzvec(v::SymTenField) =
    FluidTensors.xzvec(v.c)
@inline FluidTensors.yyvec(v::SymTenField) =
    FluidTensors.yyvec(v.c)
@inline FluidTensors.yzvec(v::SymTenField) =
    FluidTensors.yzvec(v.c)
@inline FluidTensors.zzvec(v::SymTenField) =
    FluidTensors.zzvec(v.c)

SymTenField(xx::ScalarField{T,N,N2,L},xy::ScalarField{T,N,N2,L},xz::ScalarField{T,N,N2,L},yy::ScalarField{T,N,N2,L},yz::ScalarField{T,N,N2,L},zz::ScalarField{T,N,N2,L}) where {T,N,N2,L} = SymTenField{T,N,N2,L}(xx,xy,xz,yy,yz,zz)

SymTenField{T}(dims::NTuple{3,Int},l::NTuple{3,Real}) where T = SymTenField(ScalarField{T}(dims,l),ScalarField{T}(dims,l),ScalarField{T}(dims,l),ScalarField{T}(dims,l),ScalarField{T}(dims,l),ScalarField{T}(dims,l))
SymTenField(dims::NTuple{3,Int},l::NTuple{3,Real}) = SymTenField(ScalarField(dims,l),ScalarField(dims,l),ScalarField(dims,l),ScalarField(dims,l),ScalarField(dims,l),ScalarField(dims,l))

Base.similar(a::SymTenField) = SymTenField(similar(a.c.xx),similar(a.c.xy),similar(a.c.xz),similar(a.c.yy),similar(a.c.yz),similar(a.c.zz))

function InplaceRealFFT.rfft!(v::SymTenField)
    rfft!(v.c.xx)
    rfft!(v.c.xy)
    rfft!(v.c.xz)
    rfft!(v.c.yy)
    rfft!(v.c.yz)
    rfft!(v.c.zz)
    return v
end

function InplaceRealFFT.brfft!(v::SymTenField)
    brfft!(v.c.xx)
    brfft!(v.c.xy)
    brfft!(v.c.xz)
    brfft!(v.c.yy)
    brfft!(v.c.yz)
    brfft!(v.c.zz)
    return v.r
end

function InplaceRealFFT.irfft!(v::SymTenField)
    irfft!(v.c.xx)
    irfft!(v.c.xy)
    irfft!(v.c.xz)
    irfft!(v.c.yy)
    irfft!(v.c.yz)
    irfft!(v.c.zz)
    return v.r
end

isrealspace(a::SymTenField) =
    isrealspace(a.c.xx) && isrealspace(a.c.xy) && isrealspace(a.c.xz) && isrealspace(a.c.yy) && isrealspace(a.c.yz) && isrealspace(a.c.zz)

Base.show(io::IO,m::MIME"text/plain",a::SymTenField) =
    isrealspace(a) ? show(io,m,a.r) : show(io,m,a.c)

setreal!(a::SymTenField) = (setreal!(a.c.xx); setreal!(a.c.xy); setreal!(a.c.xz); setreal!(a.c.yy); setreal!(a.c.yz); setreal!(a.c.zz))

setfourier!(a::SymTenField) = (setfourier!(a.c.xx); setfourier!(a.c.xy); setfourier!(a.c.xz); setfourier!(a.c.yy); setfourier!(a.c.yz); setfourier!(a.c.zz))

fourier!(a::SymTenField) = (fourier!(a.c.xx); fourier!(a.c.xy); fourier!(a.c.xz); fourier!(a.c.yy); fourier!(a.c.yz); fourier!(a.c.zz); a)

real!(a::SymTenField) = (real!(a.c.xx); real!(a.c.xy); real!(a.c.xz); real!(a.c.yy); real!(a.c.yz); real!(a.c.zz); a.r)

################################################################# AntiSymTenField
struct AntiSymTenField{T,N,N2,L} <: AbstractAntiSymTenArray{Complex{T},N}
    rr::AntiSymTenArray{T,N,Array{T,N},Array{T,N},Array{T,N}}
    c::AntiSymTenArray{Complex{T},N,ScalarField{T,N,N2,L},ScalarField{T,N,N2,L},ScalarField{T,N,N2,L}}
    r::SubArray{AntiSymTen{T},N,AntiSymTenArray{T,N,Array{T,N},Array{T,N},Array{T,N}},Tuple{Base.OneTo{Int},Vararg{Base.Slice{Base.OneTo{Int}},N2}},L} 

    function AntiSymTenField{T,N,N2,L}(xy::ScalarField{T,N,N2,L},xz::ScalarField{T,N,N2,L},yz::ScalarField{T,N,N2,L}) where {T,N,N2,L}
        c = AntiSymTenArray(xy,xz,yz)
        rr = AntiSymTenArray(InplaceRealFFT.data(xy),InplaceRealFFT.data(xz),InplaceRealFFT.data(yz))
        r = view(rr, Base.OneTo(size(xy.field.r, 1)), ntuple(i->Colon(), Val(N2))...)
        return new{T,N,N2,L}(rr,c,r)
    end

end

@inline function Base.getproperty(a::S,s::Symbol) where {S<:AntiSymTenField}
    if (s === :kx || s === :ky || s === :kz || s === :k)
        return getproperty(getfield(getfield(a,:c),:xy),s)
    else
        return getfield(a,s)
    end
end

@inline FluidTensors.xyvec(v::AntiSymTenField) =
    FluidTensors.xyvec(v.c)
@inline FluidTensors.xzvec(v::AntiSymTenField) =
    FluidTensors.xzvec(v.c)
@inline FluidTensors.yzvec(v::AntiSymTenField) =
    FluidTensors.yzvec(v.c)

AntiSymTenField(xy::ScalarField{T,N,N2,L},xz::ScalarField{T,N,N2,L},yz::ScalarField{T,N,N2,L}) where {T,N,N2,L} = AntiSymTenField{T,N,N2,L}(xy,xz,yz)

AntiSymTenField{T}(dims::NTuple{3,Int},l::NTuple{3,Real}) where T = AntiSymTenField(ScalarField{T}(dims,l),ScalarField{T}(dims,l),ScalarField{T}(dims,l))
AntiSymTenField(dims::NTuple{3,Int},l::NTuple{3,Real}) = AntiSymTenField(ScalarField(dims,l),ScalarField(dims,l),ScalarField(dims,l))

Base.similar(a::AntiSymTenField) = AntiSymTenField(similar(a.c.xy),similar(a.c.xz),similar(a.c.yz))

function InplaceRealFFT.rfft!(v::AntiSymTenField)
    rfft!(v.c.xy)
    rfft!(v.c.xz)
    rfft!(v.c.yz)
    return v
end

function InplaceRealFFT.brfft!(v::AntiSymTenField)
    brfft!(v.c.xy)
    brfft!(v.c.xz)
    brfft!(v.c.yz)
    return v.r
end

function InplaceRealFFT.irfft!(v::AntiSymTenField)
    irfft!(v.c.xy)
    irfft!(v.c.xz)
    irfft!(v.c.yz)
    return v.r
end

isrealspace(a::AntiSymTenField) =
    isrealspace(a.c.xy) && isrealspace(a.c.xz) && isrealspace(a.c.yz)

Base.show(io::IO,m::MIME"text/plain",a::AntiSymTenField) =
    isrealspace(a) ? show(io,m,a.r) : show(io,m,a.c)

setreal!(a::AntiSymTenField) = (setreal!(a.c.xy); setreal!(a.c.xz); setreal!(a.c.yz))

setfourier!(a::AntiSymTenField) = (setfourier!(a.c.xy); setfourier!(a.c.xz); setfourier!(a.c.yz))

fourier!(a::AntiSymTenField) = (fourier!(a.c.xy); fourier!(a.c.xz); fourier!(a.c.yz); a)

real!(a::AntiSymTenField) = (real!(a.c.xy); real!(a.c.xz); real!(a.c.yz); a.r)

###############################################

const AbstractTensorField{T} = Union{<:SymTenField{T},<:SymTrTenField{T},<:AntiSymTenField{T}}