struct ScalarField{T<:Union{Float64,Float32},N,N2,L} <: AbstractPaddedArray{T,N}
    field::PaddedArray{T,N,N2,L}
    fplan::FFTW.rFFTWPlan{T,-1,true,N}
    bplan::FFTW.rFFTWPlan{Complex{T},1,true,N}
    realspace::Base.RefValue{Bool}
    
    function ScalarField(field::PaddedArray{T,N,N2,L}) where {T,N,N2,L}
        fplan = plan_rfft!(field,1:N,flags=FFTW.MEASURE)
        bplan = plan_brfft!(field,1:N,flags=FFTW.MEASURE)
        return new{T,N,N2,L}(field,fplan,bplan,Ref(true))
    end 

end

ScalarField{T}(dims::Vararg{Int,N}) where {T,N} = ScalarField(PaddedArray{T}(dims...))
ScalarField(dims::Vararg{Int}) = ScalarField{Float64}(dims...)

Base.similar(a::ScalarField) = ScalarField(similar(a.field))

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
