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

# ╔═╡ c1543a0a-59c5-4953-855f-1804aa40a8a5
using DataFrames

# ╔═╡ 21c2750c-2882-11f1-a92f-398891230811
using EBM, CairoMakie, StatsBase

# ╔═╡ 7a45ec5c-eae6-4f8f-b079-f78b48e8f31c
@info Threads.nthreads()

# ╔═╡ fbb723e4-c656-4f06-8987-8724b0635c12
Revise.retry()

# ╔═╡ 79e6b7ee-baf0-4d37-b0ec-3e830543b925
# ╠═╡ show_logs = false
sweep = Traffic.run_all(resolution = 10, depth=100)

# ╔═╡ dc041d84-e0b0-483b-9e4f-260901d500ae
sweep_logger = Dict(k => Traffic.MeanLogger(v) for (k,v) in sweep)

# ╔═╡ 02de7688-dc32-445f-8b78-9328b873d5bc
# ╠═╡ show_logs = false
abm_sweep = Traffic.sweep_weights(Traffic.ABM(), resolution = 10, depth = 100);

# ╔═╡ f4fd8918-37cd-41ec-8e83-fa766d6df671
mean_age = Dict(k => v.mean_age for (k, v) in sweep_logger)


# ╔═╡ e04a625c-f493-4fc1-b3ab-baacc8df6595
abm_mean_age = combine(groupby(vcat([r[1] for r in abm_sweep]...),:time),:age => mean, :habitus => abs ∘ mean );

# ╔═╡ 0645842c-9a17-49e1-823f-6cd701e32e0b
generate_label(str) = replace(string(typeof(str)), "EBM.Traffic." => "")

# ╔═╡ c6507f4e-e1c1-4498-b91f-1a805c641175
begin
	f = Figure()
	ax = Axis(f[1,1], xlabel = "Step", ylabel="Mean age")
	for (strategy, vals) in mean_age
        lines!(ax,
			   vals,
			   label = generate_label(strategy))
    end
	lines!(ax, abm_mean_age.age_mean, label = "ABM", linestyle=:dot)

	Legend(f[2, 1], ax)

	save("../plots/mean_age.png",f)
f
	
end


# ╔═╡ a4cbd9d1-81a6-4d68-86a9-623da00ca7ba
mean_habitus = Dict(k => v.mean_abs_habitus for (k, v) in sweep_logger)


# ╔═╡ 53495d67-9325-4410-929d-30123e0f4f31
begin
	f2 = Figure()
	ax2 = Axis(f2[1,1], xlabel="Step", ylabel="Mean absolute habitus")
	for (strategy, vals) in mean_habitus
        lines!(ax2, vals, label = generate_label(strategy))
    end
	        lines!(ax2, abm_mean_age.habitus_abs_mean, label = "ABM", linestyle=:dot)

	Legend(f2[1, 2], ax2)

	save("../plots/habitus.png",f2)
	f2
	
end

# ╔═╡ b7e47d12-e79a-4869-bd8c-4298aeeeae81
only_hab = Dict(k => Traffic.MeanLogger(filter(v) do val
	val.weights.wₕ >= 0.8
end) for (k,v) in sweep)

# ╔═╡ 6b55052a-8de0-4e43-bc6c-50aaaf0b5297
score(sw) = mean(sw.logger.mean_age)

# ╔═╡ 06e73989-be81-4b2b-a03d-e466f9ecdf82
begin
	f4 = Figure()
	ax4 = Axis(f4[1,1])
	for (strategy, vals) in only_hab
        lines!(ax4, vals.mean_age, label = string(typeof(strategy)))
    end
	Legend(f4[1, 2], ax4)

	f4
	
end

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

# ╔═╡ 7feeff36-fc1d-4928-b59a-32d29b536263
begin
	f5 = Figure()
	ax5 = Axis(f5[1,1])
	for (strategy, vals) in sweep_logger
        lines!(ax5, vals.stay_ratio[5:end], label = string(typeof(strategy)))
    end
	Legend(f5[1, 2], ax5)

	f5
	
end

# ╔═╡ Cell order:
# ╠═e214077c-71e6-469d-abdc-0cf63260450d
# ╠═7a45ec5c-eae6-4f8f-b079-f78b48e8f31c
# ╠═a8931600-799e-400b-aef7-60061a02eba0
# ╠═8cbcc055-ab0c-4fe9-b4c6-82471a51af3c
# ╠═c1543a0a-59c5-4953-855f-1804aa40a8a5
# ╠═21c2750c-2882-11f1-a92f-398891230811
# ╠═fbb723e4-c656-4f06-8987-8724b0635c12
# ╠═79e6b7ee-baf0-4d37-b0ec-3e830543b925
# ╠═dc041d84-e0b0-483b-9e4f-260901d500ae
# ╠═02de7688-dc32-445f-8b78-9328b873d5bc
# ╠═f4fd8918-37cd-41ec-8e83-fa766d6df671
# ╠═e04a625c-f493-4fc1-b3ab-baacc8df6595
# ╠═0645842c-9a17-49e1-823f-6cd701e32e0b
# ╠═c6507f4e-e1c1-4498-b91f-1a805c641175
# ╠═53495d67-9325-4410-929d-30123e0f4f31
# ╟─a4cbd9d1-81a6-4d68-86a9-623da00ca7ba
# ╟─b7e47d12-e79a-4869-bd8c-4298aeeeae81
# ╟─6b55052a-8de0-4e43-bc6c-50aaaf0b5297
# ╠═06e73989-be81-4b2b-a03d-e466f9ecdf82
# ╠═2bc614f0-65c4-40cc-8ac6-32061d97ba00
# ╠═9fcad449-a5b2-4bf1-9b2a-1aa44a6587d9
# ╠═7feeff36-fc1d-4928-b59a-32d29b536263
