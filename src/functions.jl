function dealias!(out::T,u::T,d::Real) where {T<:AbstractField}
    isrealspace(u) && fourier!(u)
    isrealspace(out) && setfourier!(out)
    kx = u.kx
    ky = u.ky
    kz = u.kz
    kmax2 = (d*kx[end])^2

    @inbounds for k in axes(u,3)
        kz2 = kz[k]*kz[k]
        for j in axes(u,2)
            ky2 = ky[j]*ky[j]
            @simd for i in axes(u,1)
                kx2 = kx[i]*kx[i]
                out[i,j,k] = ((kx2<kmax2) | (ky2<kmax2) | (kz2<kmax2))*u[i,j,k]
            end
        end
    end
    
    return out
end