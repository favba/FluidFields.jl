RK_LISTF64 = Dict{Pair{Int,Float64},Vector{Float64}}()
K_LISTF64 = Dict{Pair{Int,Float64},Vector{Float64}}()

RK_LISTF32 = Dict{Pair{Int,Float32},Vector{Float32}}()
K_LISTF32 = Dict{Pair{Int,Float32},Vector{Float32}}()

LISTS = Dict(Float64 => (RK_LISTF64,K_LISTF64), Float32 => (RK_LISTF32, K_LISTF32))

struct Space{T<:Float3264}
    nx::Int
    ny::Int
    nz::Int
    lx::T # Length/π
    ly::T # Length/π
    lz::T # Length/π
    kx::Vector{T}
    ky::Vector{T}
    kz::Vector{T}
    k::VecArray{T,3,HomogeneousArray{T,3,Vector{T},1},HomogeneousArray{T,3,Vector{T},2},HomogeneousArray{T,3,Vector{T},3}}

    function Space{T}(nx::Integer,ny::Integer,nz::Integer,lx::Real,ly::Real,lz::Real) where {T<:Float3264}

        RK_LIST, K_LIST = LISTS[T]

        px = Int(nx)=>T(lx)
        if haskey(RK_LIST,px)
            kx = RK_LIST[px]
        else
            kx = T[(nx/2 - i)/lx for i = nx/2:-1:0]
            RK_LIST[px] = kx
        end

        py = Int(ny)=>T(ly)
        if haskey(K_LIST,py)
            ky = K_LIST[py]
        else
            ky = if iseven(ny)
                vcat(T[(ny/2 - i)/ly for i = ny/2:-1:1],T[-i/ly for i = ny/2:-1:1])
            else 
                vcat(T[(ny/2 - i)/ly for i = ny/2:-1:0],T[-i/ly for i = (ny-1)/2:-1:1])
            end
            K_LIST[py] = ky
        end

        pz = Int(nz)=>T(lz)
        if haskey(K_LIST,pz)
            kz = K_LIST[pz]
        else
            kz = if iseven(nz)
                vcat(T[(nz/2 - i)/lz for i = nz/2:-1:1],T[-i/lz for i = nz/2:-1:1])
            else 
                vcat(T[(nz/2 - i)/lz for i = nz/2:-1:0],T[-i/lz for i = (nz-1)/2:-1:1])
            end
            K_LIST[pz] = kz
        end

        s = (nx÷2 + 1,ny,nz)
        k = VecArray(HomogeneousArray{1}(kx,s),
                     HomogeneousArray{2}(ky,s),
                     HomogeneousArray{3}(kz,s))

        return new{T}(nx,ny,nz,lx,ly,lz,kx,ky,kz,k)
    end
end