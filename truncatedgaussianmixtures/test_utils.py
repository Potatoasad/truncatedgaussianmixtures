from .julia_import import jl

jl.seval("""
using TruncatedGaussianMixtures, Distributions, DataFrames
function generate_random_covariance_matrix(;d=2, K=3, cov=:full, diag_scale_low=0.5, diag_scale_high=6.0)
	diag_scaling = rand(diag_scale_low:0.1:diag_scale_high)
	rand_rotation = qr(randn(d,d)).R
	if cov == :diag
		rand_rotation = one(rand_rotation)
	end
	Σ = rand_rotation' * diagm(rand(d).*diag_scaling.*1.0) * rand_rotation;
	Σ .= (Σ + Σ')/2
	Σ
end
""")


def generate_random_mixture(d=2, K=3, cov="full", diag_scale_low=0.5, diag_scale_high=6.0):
	cov_matrix = jl.generate_random_covariance_matrix(d=2, K=3, cov=:full, diag_scale_low=0.5, diag_scale_high=6.0)