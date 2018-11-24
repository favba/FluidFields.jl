abstract type AbstractKvec{T<:AbstractFloat} <: AbstractVector{T} end

Base.IndexStyle(::Type{<:AbstractKvec}) = IndexLinear()
@inline Base.length(a::AbstractKvec) = a.n
@inline Base.size(a::AbstractKvec) = (a.n,)

struct RKvec{T<:AbstractFloat} <: AbstractKvec{T}
    n::Int
    l::T
    @inline function RKvec{T}(nx::Integer,lx::Real) where {T}
        n = div(nx,2) + 1
        return new{T}(n,T(lx))
    end
end

@inline function Base.getindex(a::RKvec{T},i::Integer) where {T}
    l = a.l
    n = a.n
    @boundscheck checkbounds(a,i)
    return (i-1)/l
end

struct Kvec{T<:AbstractFloat} <: AbstractKvec{T}
    n::Int
    l::T
    p::Int
    @inline function Kvec{T}(n::Integer,l::Real) where {T}
        p = (n+1)÷2
        return new{T}(n,T(l),p)
    end
end

@inline function Base.getindex(a::Kvec{T},i::Integer) where {T}
    l = a.l
    n = a.n
    p = a.p
    @boundscheck checkbounds(a,i)
    no2 = (i <= p)
    return ifelse(no2, (i-1)/l, (i-n-1)/l)
end
