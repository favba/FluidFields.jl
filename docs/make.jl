using Documenter, FluidFields

makedocs(
    modules = [FluidFields],
    format = :html,
    sitename = "FluidFields.jl",
    pages = Any["index.md"]
)

deploydocs(
    repo = "github.com/favba/FluidFields.jl.git",
    target = "build",
    julia = "1.0",
    deps = nothing,
    make = nothing,
)
