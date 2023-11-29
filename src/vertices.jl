# Defining the vertex structure and the on it.
#module vertices

import LinearAlgebra ,Base, Statistics
#export Vertex, subVertex, addVertex, lambdaVertex, negVertex, xangle, fetchX, fetchY, distVertex


"""
    Vertex

Represents a vertex in a geometric context, characterized by its coordinates.

# Fields
- `v::Vector{Real}`: A vector of real numbers representing the coordinates of the vertex.

# Example
```julia
vertex = Vertex([1.0, 2.0, 3.0])  # A vertex in 3-dimensional space
"""
mutable struct Vertex
    v::Vector{Real}
end

# we define functions as well as overload the + and - operators from Base

function subVertex(v1::Vertex,v2::Vertex)
    # substraction 
    return  Vertex(v1.v-v2.v)
end

Base.:(-)(v1::Vertex,v2::Vertex) = Vertex(v1.v-v2.v)

function addVertex(v1::Vertex,v2::Vertex)
    # summation
    return Vertex(v1.v+v2.v)
end

Base.:(+)(v1::Vertex,v2::Vertex) = Vertex(v1.v+v2.v)

function lambdaVertex(λ::Real,ver::Vertex)
    # a constant times a vertex
    return Vertex(λ*ver.v)
end

function lambdaVertex(ver::Vertex,λ::Real)
    # a vertex times a constant
    return Vertex(λ*ver.v)
end
    
Base.:(*)(λ::Real,ver::Vertex) = lambdaVertex(λ,ver)
Base.:(*)(ver::Vertex,λ::Real) = lambdaVertex(ver,λ)

function negVertex(ver::Vertex)
    # negating a vertex
    return Vertex(-ver.v)
end

Base.:(-)(ver::Vertex) = negVertex(ver)

function dot(v1::Vertex,v2::Vertex)
    # dot product of two vertices
    return LinearAlgebra.dot(v1.v,v2.v)
end

function norm(ver::Vertex)
    # the norm of a vertex
    return LinearAlgebra.norm(ver.v)
end

LinearAlgebra.:(*)(v1::Vertex,v2::Vertex) = dot(v1::Vertex,v2::Vertex)

function xangle(p1::Vertex,p2::Vertex)
    #computes the angle that the vector starting from vertex p1 and ending at vertex p2 makes with the x-axis
    Δ = (p2-p1).v
    flag=false
    if Δ[2] < 0
        Δ[2] = -Δ[2]
        flag = true
    end
    xang =atan(Δ[2],abs(Δ[1]))
    if Δ[1]<0
        xang = π-xang
    end
    if flag
        xang=2*π-xang
    end
    return xang

end

function fetchY(ver::Vertex)
    # taking the y-coordinate out of the vertex
    return ver.v[2]
end

function fetchX(ver::Vertex)
    # taking the x-coordinate out of the vertex
    return ver.v[1]
end

 
function L1dist(v1::Vertex,v2::Vertex)
    v = v1-v2
    return maximum(abs.(v.v))
end

function L2dist(v1::Vertex,v2::Vertex)
    v = v1-v2
    return LinearAlgebra.norm(v.v)
end

Statistics.sum(vec::Vector{Vertex}) = Vertex(sum(vec[i].v for i=1:length(vec)))


#end #of module vertices