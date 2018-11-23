abstract type AbstractKvec{T<:AbstractFloat} <: AbstractVector{T} end

Base.IndexStyle(::Type{<:AbstractKvec}) = IndexLinear()
@inline Base.length(a::AbstractKvec) = a.n
@inline Base.size(a::AbstractKvec) = (a.n,)

struct RKvec{T<:AbstractFloat} <: AbstractKvec{T}
    n::Int
    l::T
    @inline function RKvec{T}(nx::Int,lx::T) where {T}
        n = div(nx,2) + 1
        return new{T}(n,lx)
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
end

@inline function Base.getindex(a::Kvec{T},i::Integer) where {T}
    l = a.l
    n = a.n
    @boundscheck checkbounds(a,i)
    no2 = (i <= (n+1)÷2)
    return ifelse(no2, (i-1)/l, (i-n-1)/l)
end
