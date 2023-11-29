# The segment structure
#module segments

#export Segment, dotDist, xangle 

"""
    mutable struct Segment

Represents a segment in space defined by two end vertices (`p1` and `p2`). This structure also includes functions to check the validity of the input, compute the segment's length, and determine its dimensionality.

# Fields
- `p1::Vertex`: The first vertex (endpoint) of the segment.
- `p2::Vertex`: The second vertex (endpoint) of the segment.
- `checkInput::Function`: A function to check if `p1` and `p2` have the same dimensionality.
- `length::Function`: A function to compute the length of the segment.
- `dim::Function`: A function to return the dimensionality of the segment if `p1` and `p2` have the same dimensionality; otherwise, it returns `false`.

# Constructor
- `Segment(p1, p2)`: Creates a new `Segment` object given two vertices `p1` and `p2`.

# Details
- The `checkInput` function ensures that both vertices are in the same dimensional space.
- The `length` function calculates the Euclidean distance (norm) between `p1` and `p2`.
- The `dim` function returns the dimensionality of the vertices if they are consistent, otherwise returns `false`.
"""
mutable struct Segment
    p1::Vertex
    p2::Vertex
    checkInput::Function
    length::Function
    dim::Function

    
    function Segment(p1,p2)
        this = new()

        this.p1 = p1
        this.p2 = p2

        this.checkInput = function()
            return size(this.p1.v) == size(this.p2.v)       
        end

        this.length = function()
            return norm(this.p1-this.p2)
        end

        this.dim = function()
            if this.checkInput()
                return length(this.p1.v)
            else
                return false
            end
        end

        return this        
    end
end

function dotDist(p::Vector{<:Real}, segment::Segment) 
    if segment.checkInput()
        if length(p) == segment.dim()
            p1_p2 = segment.p1 -segment.p2
            p_p2 = p -segment.p2

            λ = LinearAlgebra.dot(p1_p2,p_p2)/LinearAlgebra.dot(p1_p2,p1_p2)
            λ = max(min(λ,1),0)

            p0 = λ*segment.p1 + (1-λ)*segment.p2 

            return LinearAlgebra.norm(p-p0)
        else
            return "dimention of p doesnt match dimention of segment"
        end
    else
        return "Segment has wrong dimentions"
    end
end

function dotDist(p::Vertex, segment::Segment) 
    if segment.checkInput()
        if length(p.v) == segment.dim()
            p1_p2 = segment.p1 -segment.p2
            p_p2 = p -segment.p2

            λ = LinearAlgebra.dot(p1_p2.v,p_p2.v)/LinearAlgebra.dot(p1_p2.v,p1_p2.v)
            λ = max(min(λ,1),0)

            p0 = λ*segment.p1 + (1-λ)*segment.p2 

            return LinearAlgebra.norm((p-p0).v)
        else
            return "dimention of p doesnt match dimention of segment"
        end
    else
        return "Segment has wrong dimentions"
    end
end

function xangle(seg::Segment)
    Δ = seg.p2-seg.p1
    flag=false
    if Δ[2] < 0
        Δ[2] = -Δ[2]
        flag = true
    end
    xang =atan(Δ[2],abs(Δ[1]))
    if Δ[1]<0
        xang = pi-xang
    end
    if flag
        xang=2*pi-xang
    end
    return xang

end

#end #of module segments