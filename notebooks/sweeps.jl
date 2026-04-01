### A Pluto.jl notebook ###
# v0.20.24

using Markdown
using InteractiveUtils

# ╔═╡ e214077c-71e6-469d-abdc-0cf63260450d
using Pkg

# ╔═╡ a8931600-799e-400b-aef7-60061a02eba0
Pkg.activate("..")

# ╔═╡ 8cbcc055-ab0c-4fe9-b4c6-82471a51af3c
using Revise

# ╔═╡ 21c2750c-2882-11f1-a92f-398891230811
using EBM, CairoMakie

# ╔═╡ 79e6b7ee-baf0-4d37-b0ec-3e830543b925
sweep = Traffic.run_all()

# ╔═╡ dc041d84-e0b0-483b-9e4f-260901d500ae
sweep_logger = Dict(k => Traffic.MeanLogger(v) for (k,v) in sweep)

# ╔═╡ 83c40302-6942-4d4c-aeff-57fe888427d0
Dict( k=> sum(v.deaths) for (k,v) in sweep_logger)

# ╔═╡ 7b12826f-df23-4eca-bcf3-7ff79e962a5f
sweep

# ╔═╡ Cell order:
# ╠═e214077c-71e6-469d-abdc-0cf63260450d
# ╠═a8931600-799e-400b-aef7-60061a02eba0
# ╠═8cbcc055-ab0c-4fe9-b4c6-82471a51af3c
# ╠═21c2750c-2882-11f1-a92f-398891230811
# ╠═79e6b7ee-baf0-4d37-b0ec-3e830543b925
# ╠═dc041d84-e0b0-483b-9e4f-260901d500ae
# ╠═83c40302-6942-4d4c-aeff-57fe888427d0
# ╠═7b12826f-df23-4eca-bcf3-7ff79e962a5f
