module SetBLP

import LinearAlgebra ,Base
using Statistics, Random, Distributions
using DataFrames,Plots

####################
###   Includes:  ###
####################

include("vertices.jl") 
export Vertex, subVertex, addVertex, lambdaVertex, negVertex #, xangle, fetchX, fetchY

include("segments.jl") 
export Segment, dotDist #, xangle

include("polygons.jl")  
export Polygon, minkowskiSum, lambdaPolygon, dirHausdorff, hausdorff, sortPolygon!, angles, scatterPolygon, plotPolygon

###############################
###   Defined Structures:   ###
###############################  

"""
    Options

Represents configuration options for a Monte Carlo simulation or a similar computational process.

# Fields
- `MC_iterations::Int64`: The number of iterations to be performed in the Monte Carlo simulation.
- `seed::Int64`: The seed value for random number generation, ensuring reproducibility.
- `rng::AbstractRNG`: The random number generator to be used. It should be a subtype of `AbstractRNG`.
- `conf_level::Float64`: The confidence level for statistical calculations, typically between 0 and 1.
- `tol::Float64`: The tolerance level for convergence or accuracy in computations.

# Example
```julia
options = Options(10000, 1234, MersenneTwister(), 0.95, 0.01)
"""
mutable struct Options
	MC_iterations::Int64
	seed::Int64
	rng::AbstractRNG
	conf_level::Float64
	tol::Float64
end

function Base.show(o::Options; io::IO=stdout)
	println(io, "Options:")
	println(io, "  Number of MC iterations: ", o.MC_iterations)
	println(io, "  Seed: ", o.seed)
	println(io, "  Random Number Generator: ", o.rng)
	println(io, "  Confidence level: ", o.conf_level)
	println(io, "  Tolerance level: ",o.tol)
  end

"""
  TestResults

Represents the results of a statistical test, including confidence intervals, critical values, and test statistics.

# Fields
- `ConfidenceInterval::Union{Vector{<:Real}, Nothing}`: A vector of real numbers representing the confidence interval of the test. This field is necessary.
- `criticalVal::Union{<:Real, Nothing}`: A real number representing the critical value of the test. This field is optional.
- `testStat::Union{<:Real, Nothing}`: A real number representing the test statistic. This field is optional.

# Example
```julia
results = TestResults([2.5, 3.5], 1.96, 2.3)
"""
mutable struct TestResults
	ConfidenceInterval :: Union{Vector{<:Real},Nothing} #value is necessary
	criticalVal :: Union{<:Real,Nothing} #value is optional
	testStat :: Union{<:Real,Nothing} #value is optional
end

"""
    Results

Represents the results of a statistical analysis, including bounds, null values, hypothesis test results, and derivative hypothesis test results.

# Fields
- `bound::Vector{<:Real}`: A vector of real numbers representing the bounds. This field must have a value.
- `null::Union{Vector{<:Real}, Nothing}`: A vector of real numbers representing null values. This field is optional.
- `Htest::Union{TestResults, Nothing}`: An instance of `TestResults` representing the results of a hypothesis test. This field is optional.
- `dHtest::Union{TestResults, Nothing}`: An instance of `TestResults` representing the results of a derivative hypothesis test. This field is optional.

# Example
```julia
bounds = [1.0, 2.0]
null_vals = [0.0, 0.1]
htest = TestResults([2.5, 3.5], 1.96, 2.3)
results = Results(bounds, null_vals, htest, nothing)
"""
mutable struct Results
	bound :: Vector{<:Real} 			   # This field must have a value
	null  :: Union{Vector{<:Real},Nothing} # Value is optional
	Htest :: Union{TestResults,Nothing}    # Value is optional
	dHtest :: Union{TestResults,Nothing}   # Value is optional
end

function Base.show(r::Results; io::IO=stdout, digits::Int=4)
    print(io, "Results: \n")
    print(io, "  Null: $(round.(r.null, digits=digits))\n") 
    print(io, "  Bound: $(round.(r.bound, digits=digits))\n") 
    print(io, "  Hausdorff based test: \n")
    print(io, "    Test Stat: $(round(r.Htest.testStat, digits=digits))\n")
    print(io, "    Critical Value: $(round(r.Htest.criticalVal, digits=digits))\n")
    print(io, "    Confidence Interval: $(round.(r.Htest.ConfidenceInterval, digits=digits))\n")
    print(io, "  directed Hausdorff test: \n")
    print(io, "    Test Stat: $(round(r.dHtest.testStat, digits=digits))\n")
    print(io, "    Critical Value: $(round(r.dHtest.criticalVal, digits=digits))\n")
    print(io, "    Confidence Interval: $(round.(r.dHtest.ConfidenceInterval, digits=digits))\n")
end

#####################
###   Constants   ###
#####################

const default_options = Options(2000,15217,MersenneTwister(),0.95,10^-6)


#####################
###  Functions:   ###
#####################

plus(x::Real)=max(0.0,x)
minus(x::Real)=max(0.0,-x)

function HdistInterval(v1::Vector{<:Real},v2::Vector{<:Real})
    v = v1 - v2
    return maximum(abs.(v))
end
  
function dHdistInterval(v1::Vector{<:Real},v2::Vector{<:Real})
    v = v1 - v2
	return maximum([plus(v[1]),minus(v[2])])
end

## Plan: add DataFrame capabilities

"""
    EY(yl::Vector{<:Real}, yu::Vector{<:Real}, H0::Vector{<:Real}; options::Options = default_options, method::String = "Asymptotic")

Calculate the expected value `EY` based on lower and upper bounds `yl` and `yu`, and a hypothesis vector `H0`. 

The function allows for different methods of computation, specifically 'Asymptotic' and 'Bootstrap' methods.

# Arguments
- `yl::Vector{<:Real}`: Vector of lower bounds.
- `yu::Vector{<:Real}`: Vector of upper bounds.
- `H0::Vector{<:Real}`: Hypothesis vector.
- `options::Options`: Configuration options for the calculation, defaults to `default_options`.
- `method::String`: Method of computation, either "Asymptotic" or "Bootstrap", defaults to "Asymptotic".

# Returns
Depends on the method chosen:
- If `method` is "Asymptotic", calls `EYasy`.
- If `method` is "Bootstrap", calls `EYboot`.

# Examples
```julia
# Example using the Asymptotic method
result = EY([1.0, 2.0], [3.0, 4.0], [0.5, 0.5])

# Example using the Bootstrap method
result = EY([1.0, 2.0], [3.0, 4.0], [0.5, 0.5], method="Bootstrap")
"""
function EY(yl::Vector{<:Real},yu::Vector{<:Real},H0::Vector{<:Real};options::Options=default_options,method="Asymptotic")
	#THis is the shell function that calls either the asymtotic distribution version or the bootstrap version of EY
	if method =="Asymptotic"
		EYasy(yl,yu,H0,options)
	else
		EYboot(yl,yu,H0,options)
	end
end


function EYboot(yl::Vector{<:Real},yu::Vector{<:Real},H0::Vector{<:Real},options::Options=default_options)
	#This function uses a bootstrap test. This option is not in BM(2008) for EY but it is proved for BLP in section 4
	LB = mean(yl)
	UB = mean(yu)
	bound = [LB,UB]

	# test Statistic
	n = length(yl) 
	sqrt_n = sqrt(n)
	bound = vec(bound)
	testStat_H = sqrt_n*HdistInterval(bound,H0)
	testStat_dH = sqrt_n*dHdistInterval(bound,H0)

	B = options.MC_iterations #number of MC iterations to compute the critical value
	α = options.conf_level  #confidence level for the critical value1
	distribution = DiscreteUniform(1,n)

	r_H=zeros(B)
	r_dH = zeros(B)

	for i=1:B
		indx = rand(options.rng,distribution,n)
		yl_b = yl[indx]
		yu_b = yu[indx]
		bound_b = [mean(yl_b),mean(yu_b)]
		r_H[i] = sqrt_n * HdistInterval(bound_b,bound)
		r_dH[i] = sqrt_n * dHdistInterval(bound_b,bound)
	end
	sort!(r_H)
	c_H = r_H[floor(Int64,α*B)]
	CI_H = [LB-c_H/sqrt_n,UB+c_H/sqrt_n]
	Htest = TestResults(CI_H,c_H,testStat_H) 

	sort!(r_dH)
	c_dH = r_dH[floor(Int64,α*B)]
	CI_dH = [LB-c_dH/sqrt_n,UB+c_dH/sqrt_n]
	dHtest = TestResults(CI_dH,c_dH,testStat_dH)
	#TestResults(CI_dH,c_dH,testStat_dH)

	results = Results(bound,H0,Htest,dHtest)

	return results
end

function EYasy(yl::Vector{<:Real},yu::Vector{<:Real},H0::Vector{<:Real},options::Options=default_options)

	Random.seed!(options.seed)
	#This function uses the test based on the asymptotic distributin as developed in BM(2008) pp. 778-779
    LB = mean(yl)
	UB = mean(yu)
	bound = [LB,UB]

	# test Statistic
	n = length(yl)
	sqrt_n = sqrt(n)
	testStat_H = sqrt_n*HdistInterval(bound,H0)
	testStat_dH = sqrt_n*dHdistInterval(bound,H0)

	#Simulating the asy. distribution using a MC method to establish a critical value (quantile):

	# Drawing pairs of bivariate normal r.v.'s 
	σ = cov(yl,yu)
	Pi = [var(yl) σ; σ var(yu)] #covariance matrix for yl,yu
	
	d = MvNormal([0, 0],Pi) #defining the joint normal distribution
	B = options.MC_iterations #number of MC iterations to compute the critical value
	α = options.conf_level  #confidence level for the critical value1

	## Following Algorithm on page 780 in BM2008:
	rr = (rand(d,B)); #drawing B pairs from a bivariate-normal distribution.
	
	## test based on Hausdorff distance:
	r_H = maximum(abs.(rr),dims=1);
	sort!(r_H,dims=2)
	c_H = r_H[floor(Int64,α*B)]
	CI_H = [LB-c_H/sqrt_n,UB+c_H/sqrt_n]
	Htest = TestResults(CI_H,c_H,testStat_H) 

	#test based on directed Hausdorff distance:
	r_dH = maximum([plus.(rr[1,:]) minus.(rr[2,:])],dims=2)
	sort!(r_dH,dims=1)
	c_dH = r_dH[floor(Int64,α*B)]
	CI_dH = [LB-c_dH/sqrt_n,UB+c_dH/sqrt_n]
	dHtest = TestResults(CI_dH,c_dH,testStat_dH)

	results = Results(bound,H0,Htest,dHtest)

	return results
end

#############################
###  projection functions ###
#############################

## Vector/Matrix versions ##

# 1. x is assumed to be one dimensional

"""
    projection(yl::Vector{<:Real}, yu::Vector{<:Real}, x::Vector{<:Real})

Compute the projection bounds for a given vector `x` within the lower and upper bounds specified by `yl` and `yu`.

This function demeans `x`, computes element-wise products of `x` with `yl` and `yu`, and then calculates the sum of the minimum and maximum of these products normalized by the sum of squares of `x`.

# Arguments
- `yl::Vector{<:Real}`: A vector of lower bounds.
- `yu::Vector{<:Real}`: A vector of upper bounds.
- `x::Vector{<:Real}`: The vector for which projection bounds are to be calculated.

# Returns
- `Vector{Real}`: A 2-element vector containing the lower and upper projection bounds `[lb, ub]`.
"""
function projection(yl::Vector{<:Real},yu::Vector{<:Real},x::Vector{<:Real})
	x = x.-mean(x) #demean x
	M = [x.*yl x.*yu]
	s = sum(x.*x)
	lb = sum(minimum(M,dims=2)) / s
	ub = sum(maximum(M,dims=2)) / s 
	return [lb ub]
end

# 2. X is assumed to be a matrix of covariates and a single coordinate is specified

"""
    projection(yl::Vector{<:Real}, yu::Vector{<:Real}, x::Matrix{<:Real}, cord::Int64)

Compute the projection bounds for a specific column (indicated by `cord`) of the matrix `x`, considering the lower and upper bounds `yl` and `yu`.

This function operates on the column specified by `cord` in matrix `x`. It first isolates this column, replaces it with a vector of ones in a copy of `x`, and then computes a prediction vector based on the modified matrix. The function then calculates the residual vector and subsequently computes its projection bounds using the `yl` and `yu` vectors.

# Arguments
- `yl::Vector{<:Real}`: A vector of lower bounds.
- `yu::Vector{<:Real}`: A vector of upper bounds.
- `x::Matrix{<:Real}`: The matrix for which the projection is calculated.
- `cord::Int64`: The column index in `x` for which the projection bounds are computed.

# Returns
- `Vector{Real}`: A 2-element vector containing the lower and upper projection bounds for the specified column in `x`.
"""
function projection(yl::Vector{<:Real},yu::Vector{<:Real},x::Matrix{<:Real},cord::Int64)
    # The function assumes that the matrix x does not contain a 1 Vector
    our_x = x[:,cord] #taking out the coordinate of interest
    new_x = copy(x)
    new_x[:,cord] .= 1.0  #replacing the column with a vector of ones
    pred_x =new_x*(inv(new_x'*new_x)*new_x'*our_x)
	res_x = our_x - pred_x
    bound = projection(yl,yu,res_x)
    return bound
end

# 3. X is assumed to be a matrix of covariates and a vector of coordinates is specified

"""
    projection(yl::Vector{<:Real}, yu::Vector{<:Real}, x::Matrix{<:Real}, cords::Vector{Int64})

Calculate the projection bounds for specific columns of the matrix `x`, identified by the indices in `cords`, using the lower and upper bounds `yl` and `yu`.

This function iterates over the column indices provided in `cords`. For each column index, it isolates the corresponding column from `x`, creates a modified version of `x` where this column is replaced with a vector of ones, and then computes a prediction vector. The function calculates the residual for this column and then computes its projection bounds using `yl` and `yu`.

# Arguments
- `yl::Vector{<:Real}`: A vector of lower bounds.
- `yu::Vector{<:Real}`: A vector of upper bounds.
- `x::Matrix{<:Real}`: The matrix for which projections are calculated.
- `cords::Vector{Int64}`: A vector of column indices in `x` for which the projection bounds are computed.

# Returns
- `Vector{Vector{Real}}`: A vector of 2-element vectors, each containing the lower and upper projection bounds for the specified columns in `x`.
"""
function projection(yl::Vector{<:Real},yu::Vector{<:Real},x::Matrix{<:Real},cords::Vector{Int64})
    # The function assumes that the matrix x does not contain a 1 Vector
	bounds = []
    for cord in cords
		our_x = x[:,cord]
		new_x = copy(x)
		new_x[:,cord] .= 1.0  #replacing the column with a vector of ones
		pred_x =new_x*(inv(new_x'*new_x)*new_x'*our_x)
		res_x = our_x - pred_x
    	bound = projection(yl,yu,res_x)
		push!(bounds,bound)
	end
    return bound
end

# 4. X is assumed to be a matrix of covariates and a coordinate is NOT specified


"""
    projection(yl::Vector{<:Real},yu::Vector{<:Real},x::Matrix{<:Real})


Calculate the projection bounds for each column of `x`, using lower and upper bounds `yl` and `yu`.

# Arguments
- `yl::Vector{<:Real}`: Lower bounds for the projection.
- `yu::Vector{<:Real}`: Upper bounds for the projection.
- `x::Matrix{<:Real}`: Matrix for which projections are calculated.

# Returns
- `Vector`: A vector containing the projection bounds for each column in `x`.
"""
function projection(yl::Vector{<:Real},yu::Vector{<:Real},x::Matrix{<:Real})
    # The function assumes that the matrix x does not contain a 1 Vector
	ncols = size(x,2)
	bounds = []
    for cord in 1:ncols
		our_x = x[:,cord]
		new_x = copy(x)
		new_x[:,cord] .= 1.0  #replacing the column with a vector of ones
		pred_x =new_x*(inv(new_x'*new_x)*new_x'*our_x)
		res_x = our_x - pred_x
    	bound = projection(yl,yu,res_x)
		push!(bounds,bound)
	end
    return bounds
end

## Data frame versions ##

"""
    projection(df::DataFrame, yl::Symbol, yu::Symbol, x::Symbol)

Calculate the projection bound for the column `x` of DataFrame `df`, using the columns `yl` and `yu` as bounds.

# Arguments
- `df::DataFrame`: DataFrame containing the data.
- `yl::Symbol`: Symbol representing the column name for lower bounds.
- `yu::Symbol`: Symbol representing the column name for upper bounds.
- `x::Symbol`: Symbol representing the column name for which projection is calculated.

# Returns
- Projection bound for each column in `x`.
"""
function projection(df::DataFrame, yl::Symbol,yu::Symbol,x::Symbol)
	y_l = copy(df[!,yl])
	y_u = copy(df[!,yu])
	new_x = Vector(df[!,x])
	bound = projection(y_l,y_u,new_x)
	return bound
end

"""
    projection(df::DataFrame, yl::Symbol, yu::Symbol, x::Symbol)

Calculate the projection bound for the column `x` of DataFrame `df`, using the columns `yl` and `yu` as bounds.

# Arguments
- `df::DataFrame`: DataFrame containing the data.
- `yl::Symbol`: Symbol representing the column name for lower bounds.
- `yu::Symbol`: Symbol representing the column name for upper bounds.
- `x::Symbol`: Symbol representing the column name for which projection is calculated.

# Returns
- Projection bound for each column in `x`.
"""
function projection(df::DataFrame, yl::Symbol,yu::Symbol,x::Vector{Symbol})
	y_l = copy(df[!,yl])
	y_u = copy(df[!,yu])
	new_x = Matrix(df[!,x])
	bounds = projection(y_l,y_u,new_x)
	return bounds
end

"""
    projection(df::DataFrame, yl::Symbol, yu::Symbol, x::Symbol)

Calculate the projection bound for the column `x` of DataFrame `df`, using the columns `yl` and `yu` as bounds.

# Arguments
- `df::DataFrame`: DataFrame containing the data.
- `yl::Symbol`: Symbol representing the column name for lower bounds.
- `yu::Symbol`: Symbol representing the column name for upper bounds.
- `x::Symbol`: Symbol representing the column name for which projection is calculated.

# Returns
- Projection bound for a specific coordinate.
"""
function projection(df::DataFrame, yl::Symbol,yu::Symbol,x::Vector{Symbol},cord::Int64)
	y_l = copy(df[!,yl])
	y_u = copy(df[!,yu])
	new_x = Matrix(df[!,x])
	bounds = projection(y_l,y_u,new_x,cord)
	return bounds
end

###################### End of projection functions ######################################

"""
    oneDproj(yl::Vector{<:Real}, yu::Vector{<:Real}, x::Vector{<:Real}; 
             options::Options=default_options, CI=true, 
             H0::Union{Vector{<:Real}, Nothing}=nothing)

Compute the one-dimensional (1D) projection of the identification set on a specific dimension of the explanatory variable `x`. The function uses lower and upper bounds `yl` and `yu`, and can optionally perform bootstrap iterations for confidence interval estimation and hypothesis testing.

# Arguments
- `yl::Vector{<:Real}`: A vector of lower bounds.
- `yu::Vector{<:Real}`: A vector of upper bounds.
- `x::Vector{<:Real}`: The explanatory variable vector.

# Optional Arguments
- `options::Options`: Configuration options for the bootstrap iterations and other parameters (defaults to `default_options`).
- `CI::Bool`: A boolean to indicate whether to compute confidence intervals (defaults to `true`).
- `H0::Union{Vector{<:Real}, Nothing}`: A vector for hypothesis testing or `nothing` if not applicable.

# Returns
- `Results`: An object containing the computed bounds, optional hypothesis test results, and confidence intervals if requested.

# Details
The function performs the following steps:
1. Compute bounds using the `projection` function.
2. If `CI` is true, perform bootstrap iterations to estimate confidence intervals and, if provided, test against the hypothesis `H0`.
3. Return the results including point estimates, confidence intervals, and test statistics.
"""
function oneDproj(yl::Vector{<:Real},
			  yu::Vector{<:Real},
			  x::Vector{<:Real};
			  options::Options=default_options, 
			  CI=true,
			  H0::Union{Vector{<:Real},Nothing}=nothing)
	## computes the 1D projection of the identification set on a specific dimesion of the explanatory variable

	#step 1: Compute the bounds on page 787 in BM2008
	
	bound = vec(projection(yl,yu,x)) #vectorizing is necessary because the HdistInterval function wants two vectrs as input
	LB = bound[1]
	UB = bound[2]

	if !CI  # all we want is a point estimator
		results = Results(bound,nothing,nothing,nothing)
		return results
	else

		#step 2: Bootstrap iterationss 
		n = length(yl)
		sqrt_n = sqrt(n)

		B = options.MC_iterations #number of MC iterations to compute the critical value
		α = options.conf_level  #confidence level for the critical value1
		distribution = DiscreteUniform(1,n)

		r_H=zeros(B)
		r_dH = zeros(B)

		for i=1:B
			indx = rand(options.rng,distribution,n)
			yl_b = yl[indx]
			yu_b = yu[indx]
			x_b  = x[indx]
			bound_b = vec(projection(yl_b,yu_b,x_b))
			r_H[i] = sqrt_n * HdistInterval(bound_b,bound)
			r_dH[i] = yqrt_n * dHdistInterval(bound_b,bound)
		end

		sort!(r_H)
		c_H = r_H[floor(Int64,α*B)]
		CI_H = [LB-c_H/sqrt_n,UB+c_H/sqrt_n]

		sort!(r_dH)
		c_dH = r_dH[floor(Int64,α*B)]
		CI_dH = [LB-c_dH/sqrt_n,UB+c_dH/sqrt_n]

		if isnothing(H0) # no testing
			Htest = TestResults(CI_H,c_H,nothing) 
			dHtest = TestResults(CI_dH,c_dH,nothing)

			results = Results(bound,nothing,Htest,dHtest)
			return results

		else	
			#step 3: Compute the test statisticss
			testStat_H = sqrt_n*HdistInterval(bound,H0)
			testStat_dH = sqrt_n*dHdistInterval(bound,H0)

			Htest = TestResults(CI_H,c_H,testStat_H) 
			dHtest = TestResults(CI_dH,c_dH,testStat_dH)

			results = Results(bound,H0,Htest,dHtest)
			return results

		end

	end

end
###########################
###  Export Statement:  ###
###########################

export Options,default_options, Results, TestResults, EY, oneDproj, projection

end #of module
