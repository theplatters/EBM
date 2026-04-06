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
using EBM, CairoMakie, StatsBase

# ╔═╡ 7a45ec5c-eae6-4f8f-b079-f78b48e8f31c
@info Threads.nthreads()

# ╔═╡ fbb723e4-c656-4f06-8987-8724b0635c12
Revise.retry()

# ╔═╡ 79e6b7ee-baf0-4d37-b0ec-3e830543b925
# ╠═╡ show_logs = false
sweep = Traffic.run_all()

# ╔═╡ dc041d84-e0b0-483b-9e4f-260901d500ae
sweep_logger = Dict(k => Traffic.MeanLogger(v) for (k,v) in sweep)

# ╔═╡ 83c40302-6942-4d4c-aeff-57fe888427d0
Dict( k=> sum(v.deaths) for (k,v) in sweep_logger)

# ╔═╡ 7b12826f-df23-4eca-bcf3-7ff79e962a5f
sweep

# ╔═╡ f4fd8918-37cd-41ec-8e83-fa766d6df671
mean_age = Dict(k => v.mean_age for (k, v) in sweep_logger)


# ╔═╡ c6507f4e-e1c1-4498-b91f-1a805c641175
begin
	f = Figure()
	ax = Axis(f[1,1])
	for (strategy, vals) in mean_age
        lines!(ax, vals, label = string(typeof(strategy)))
    end
	Legend(f[1, 2], ax)

	f
	
end


# ╔═╡ a4cbd9d1-81a6-4d68-86a9-623da00ca7ba
mean_habitus = Dict(k => v.mean_abs_habitus for (k, v) in sweep_logger)


# ╔═╡ 53495d67-9325-4410-929d-30123e0f4f31
begin
	f2 = Figure()
	ax2 = Axis(f2[1,1])
	for (strategy, vals) in mean_habitus
        lines!(ax2, vals, label = string(typeof(strategy)))
    end
	Legend(f2[1, 2], ax)

	f2
	
end

# ╔═╡ 2d2cff3e-fa94-4803-9afd-d20502bf88df
# ╠═╡ show_logs = false
 unsure_sweep = Traffic.sweep_weights(strategy=Traffic.UnsureStrategy(), depth=100, resolution=20)

# ╔═╡ 3531b0d5-b88c-4296-a30f-306a77086103
# ╠═╡ show_logs = false
 mhs_sweep = Traffic.sweep_weights(strategy=Traffic.MeanHabitusStrategy(), depth=100, resolution=20)

# ╔═╡ d79da30c-c57f-438c-9afd-9143754e7274
# ╠═╡ show_logs = false
random_sweep = Traffic.sweep_weights(strategy=Traffic.RandomStrategy(), depth=100, resolution=20)

# ╔═╡ bb33eab3-aee4-4142-8608-30ba63a713e2
# ╠═╡ show_logs = false
naive_sweep = Traffic.sweep_weights(strategy=Traffic.NaiveStrategy(), depth=100, resolution=20)

# ╔═╡ 6b55052a-8de0-4e43-bc6c-50aaaf0b5297
score(sw) = -sum(sw.logger.deaths)

# ╔═╡ 2bc614f0-65c4-40cc-8ac6-32061d97ba00
newdict = Dict(k => argmax(score, v) for (k, v) in sweep)


# ╔═╡ 9fcad449-a5b2-4bf1-9b2a-1aa44a6587d9
begin
	f3 = Figure()
	ax3 = Axis(f3[1,1])
	for (strategy, vals) in newdict
        lines!(ax3, vals.logger.mean_age, label = string(typeof(strategy)))
    end
	Legend(f3[1, 2], ax3)

	f3
	
end

# ╔═╡ b7e47d12-e79a-4869-bd8c-4298aeeeae81
only_hab = Dict(k => first(filter(v) do val
	val.weights.wₕ == 1.0
end) for (k,v) in sweep)

# ╔═╡ 06e73989-be81-4b2b-a03d-e466f9ecdf82
begin
	f4 = Figure()
	ax4 = Axis(f4[1,1])
	for (strategy, vals) in only_hab
        lines!(ax4, vals.logger.mean_age, label = string(typeof(strategy)))
    end
	Legend(f4[1, 2], ax4)

	f4
	
end

# ╔═╡ Cell order:
# ╠═e214077c-71e6-469d-abdc-0cf63260450d
# ╠═7a45ec5c-eae6-4f8f-b079-f78b48e8f31c
# ╠═a8931600-799e-400b-aef7-60061a02eba0
# ╠═8cbcc055-ab0c-4fe9-b4c6-82471a51af3c
# ╠═21c2750c-2882-11f1-a92f-398891230811
# ╠═fbb723e4-c656-4f06-8987-8724b0635c12
# ╠═79e6b7ee-baf0-4d37-b0ec-3e830543b925
# ╠═dc041d84-e0b0-483b-9e4f-260901d500ae
# ╠═83c40302-6942-4d4c-aeff-57fe888427d0
# ╠═7b12826f-df23-4eca-bcf3-7ff79e962a5f
# ╠═f4fd8918-37cd-41ec-8e83-fa766d6df671
# ╠═c6507f4e-e1c1-4498-b91f-1a805c641175
# ╠═a4cbd9d1-81a6-4d68-86a9-623da00ca7ba
# ╠═53495d67-9325-4410-929d-30123e0f4f31
# ╠═2d2cff3e-fa94-4803-9afd-d20502bf88df
# ╠═3531b0d5-b88c-4296-a30f-306a77086103
# ╠═d79da30c-c57f-438c-9afd-9143754e7274
# ╠═bb33eab3-aee4-4142-8608-30ba63a713e2
# ╠═6b55052a-8de0-4e43-bc6c-50aaaf0b5297
# ╠═2bc614f0-65c4-40cc-8ac6-32061d97ba00
# ╠═9fcad449-a5b2-4bf1-9b2a-1aa44a6587d9
# ╠═b7e47d12-e79a-4869-bd8c-4298aeeeae81
# ╠═06e73989-be81-4b2b-a03d-e466f9ecdf82
