struct ScalarField{T<:Union{Float64,Float32},N,N2,L} <: AbstractPaddedArray{T,N}
    field::PaddedArray{T,N,N2,L}
    fplan::FFTW.rFFTWPlan{T,-1,true,N}
    bplan::FFTW.rFFTWPlan{Complex{T},1,true,N}
    space::Space{T}
    realspace::Base.RefValue{Bool}
    
    function ScalarField(field::PaddedArray{T,N,N2,L},l) where {T,N,N2,L}
        fplan = plan_rfft!(field,1:N,flags=FFTW.MEASURE)
        bplan = plan_brfft!(field,1:N,flags=FFTW.MEASURE)
        space = Space{T}(size(field.r,1),size(field.r,2),size(field.r,3),l[1],l[2],l[3])
        return new{T,N,N2,L}(field,fplan,bplan,space,Ref(true))
    end 

end

@inline function Base.getproperty(a::S,s::Symbol) where {S<:ScalarField}
    if s === :rr
        return getfield(getfield(a,:field),:data)
    elseif s === :r 
        return getfield(getfield(a,:field),:r)
    elseif s === :c 
        return getfield(a,:field)
    elseif (s === :kx || s === :ky || s === :kz || s === :k)
        return getfield(getfield(a,:space),s)
    else
        return getfield(a,s)
    end
end

ScalarField{T}(dims::NTuple{3,Int},l::NTuple{3,Real}) where {T} = ScalarField(PaddedArray{T}(dims...),l)
ScalarField(dims::NTuple{3,Int},l::NTuple{3,Real}) = ScalarField{Float64}(dims,l)

function ScalarField(file,dim::NTuple{N,Int},l::Vararg{real,3}) where N
    r = ScalarField(PaddedArray(dim...),l)
    read!(file,r.field.data)
    return r
end

Base.similar(a::ScalarField) = ScalarField(similar(a.field),(a.space.lx,a.space.ly,a.space.lz))

@inline InplaceRealFFT.data(a::ScalarField) = InplaceRealFFT.data(a.field)
@inline Base.real(a::ScalarField) = real(a.field)
@inline InplaceRealFFT.complex_view(a::ScalarField) = InplaceRealFFT.complex_view(a.field)

function Base.:*(p::FFTW.rFFTWPlan{T,FFTW.FORWARD,true,N},f::ScalarField{T,N,N2,L}) where {T<:Float3264,N,N2,L} 
    isrealspace(f) || error("Field already is in Fourier space")
    mul!(InplaceRealFFT.complex_view(f),p,real(f))
    f.realspace[] = false
    f
end

InplaceRealFFT.rfft!(a::ScalarField) =
    a.fplan * a

function Base.:*(p::FFTW.rFFTWPlan{Complex{T},FFTW.BACKWARD,true,N},f::ScalarField{T,N,N2,L}) where {T<:Float3264,N,N2,L} 
    isrealspace(f) && error("Field already is in real space")
    mul!(real(f),p,InplaceRealFFT.complex_view(f))
    f.realspace[] = true
    real(f)
end

InplaceRealFFT.brfft!(a::ScalarField) =
    a.bplan * a

function InplaceRealFFT.irfft!(a::ScalarField)
    brfft!(a)
    rmul!(InplaceRealFFT.data(a),inv(prod(size(real(a)))))
    real(a)
end

isrealspace(a::ScalarField) =
    a.realspace[]

Base.show(io::IO,m::MIME"text/plain",a::ScalarField) =
    a.realspace[] ? show(io,m,real(a)) : show(io,m,a.field)

setreal!(a::ScalarField) = (a.realspace[] = true)
setfourier!(a::ScalarField) = (a.realspace[] = false)

function fourier!(a::ScalarField)
    rfft!(a)
    rmul!(InplaceRealFFT.data(a),inv(prod(size(real(a))))) 
    return a
end

real!(a::ScalarField) = brfft!(a)