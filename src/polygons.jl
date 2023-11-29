# The polygon structure and its functions.
#module Polygons

#export Polygon, minkowskiSum, lambdaPolygon, dirHausdorff, hausdorff


"""
    Polygon

Represents a geometric polygon defined by its vertices.

# Fields
- `vertices::Vector{Vertex}`: An array of `Vertex` objects defining the corners of the polygon.
- `isSorted::Bool`: Indicates whether the vertices of the polygon are sorted. It is `false` upon initialization.

# Constructor
`Polygon(vertices)` creates a new `Polygon` object with the given vertices and sets `isSorted` to `false`.

# Example
```julia
verts = [Vertex([0,0]), Vertex([1,0]), Vertex([0,1])]
poly = Polygon(verts)
"""
mutable struct Polygon
    vertices :: Vector{Vertex}
    isSorted :: Bool  #indicates whether the polygon was sorted. isSorted = false on initiation

    function Polygon(vertices)
        this = new()
        this.vertices=vertices
        this.isSorted = false     
        return this
    end
end
        
## Scatter plot function
"""
    scatterPolygon(p::Polygon)

Create a scatter plot of the vertices of a given polygon `p`. 

This function first checks if the vertices of the polygon are sorted. If not, it sorts them. It then extracts the x and y coordinates of each vertex and plots these points in a scatter plot. The plot's x and y axis limits are adjusted slightly outside the range of the polygon's vertices to ensure visibility.

# Arguments
- `p::Polygon`: A Polygon object whose vertices are to be plotted.

# Returns
- Generates a scatter plot of the polygon's vertices.

# Details
- The function assumes that the `Polygon` type has properties `isSorted` and `vertices`, where `vertices` is a list of points, each with `x` and `y` coordinates.
- If `p.isSorted` is `false`, the polygon's vertices are sorted before plotting.
- The plot automatically adjusts its axes limits to provide a margin around the plotted points for better visibility.
"""
function scatterPolygon(p::Polygon)
    if p.isSorted == false
        p.sort()
    end
    n=length(p.vertices)
    x=zeros(n); y=zeros(n);
    for i=1:n
        x[i]=p.vertices[i].v[1]
        y[i]=p.vertices[i].v[2]
    end

    mx, Mx = minimum(x), maximum(x)
    my, My = minimum(y), maximum(y)

    mx -= 0.05*abs(mx)
    Mx += 0.05*abs(Mx)

    my -= 0.05*abs(my)
    My += 0.05*abs(My)

    scatter(x,y,label="")
end

## Line plot function

"""
    plotPolygon(p::Polygon)

Plot a given polygon `p` by connecting its vertices in order. 

This function first checks if the vertices of the polygon are sorted. If not, it sorts them using `sortPolygon!`. It then extracts the x and y coordinates of each vertex and creates a closed plot by connecting these points in sequence and finally back to the first point.

The plot's x and y axis limits are adjusted slightly outside the range of the polygon's vertices to ensure all points are clearly visible within the plot area.

# Arguments
- `p::Polygon`: A Polygon object whose vertices are to be plotted.

# Returns
- Generates a plot of the polygon, with the vertices connected in order.

# Details
- The function assumes that the `Polygon` type has properties `isSorted` and `vertices`, with each vertex having `x` and `y` coordinates.
- If `p.isSorted` is `false`, the function sorts the vertices of the polygon before plotting.
- The polygon is plotted as a closed shape, with the last vertex connected back to the first.
- Axes limits are automatically adjusted to provide a margin around the polygon for better visibility.
"""
function plotPolygon(p::Polygon)
    if p.isSorted == false
        sortPolygon!(p)
    end
    n=length(p.vertices)
    x=zeros(n+1); y=zeros(n+1);
    for i=1:n
        x[i]=p.vertices[i].v[1]
        y[i]=p.vertices[i].v[2]
    end
    x[n+1]=p.vertices[1].v[1]
    y[n+1]=p.vertices[1].v[2]

    mx, Mx = minimum(x), maximum(x)
    my, My = minimum(y), maximum(y)

    mx -= 0.05*abs(mx)
    Mx += 0.05*abs(Mx)

    my -= 0.05*abs(my)
    My += 0.05*abs(My)

    plot(x,y,label="",fill=true,xlims = (mx,Mx), ylims=(my,My) )
end

## polygon angles function
function angles(p::Polygon)
    if p.isSorted == false
        sortPolygon!(p)
    end
    #this function
    n=length(p.vertices)
    ang=zeros(n)
    for i=1:n
        i==n ? j=1 : j=i+1
        ang[i] =xangle(p.vertices[i],p.vertices[j])                
    end
    return ang
end

# the y coordinate of a vertex
function fetchY(ver::Vertex)
    # taking the y-coordinate out of the vertex
    return ver.v[2]
end

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

## Sorting a Polygon function

"""
    sortPolygon!(P::Polygon)

Sort the vertices of a polygon `P` in place, arranging them in an angular order starting from the vertex with the minimal y-coordinate. 

The function first identifies the vertex with the lowest y-coordinate and makes it the first vertex of the polygon. Then, it calculates the angles between this vertex and all other vertices. Finally, it sorts the vertices in ascending order of these angles, effectively reordering the vertices around the first vertex.

This sorting method ensures that the vertices of the polygon are ordered in a way that makes them suitable for plotting or other geometric computations.

# Arguments
- `P::Polygon`: A Polygon object whose vertices are to be sorted.

# Returns
- `Polygon`: The same Polygon object `P`, with its vertices sorted in place.

# Details
- The function assumes that the `Polygon` type has a property `vertices`, with each vertex having `x` and `y` coordinates.
- The sorting is based on angular order relative to the vertex with the lowest y-coordinate, ensuring a consistent traversal order for the vertices.
- This operation modifies the original `Polygon` object `P`.
"""
function sortPolygon!(P::Polygon)
    n=length(P.vertices)
    #step 1: find the point with a minimal y coordinate and put it first.
    # comment: sorting is complexity nlog(n) but the following is just n
    #using sorting:
    #I = sortperm(fetchY.(P.vertices))
    #P.vertices = P.vertices[I]
    #going over the list
    m=fetchY(P.vertices[1])
    for i=2:n
        l=fetchY(P.vertices[i])
        if l<m #then swap
            m=l
            temp=P.vertices[i]
            P.vertices[i]=P.vertices[1]
            P.vertices[1]=temp
        end
    end
    #step 2: compute angles between the minimal vertex and all other vertices
    angs =zeros(n) #first column for angles and second column for the x coordinate
    angs[1]=-1
    v1 =P.vertices[1]
    for i=2:n
        angs[i] = xangle(v1,P.vertices[i])
    end
    #step 3: sort by angle
    I=sortperm(angs)
    P.vertices=P.vertices[I] 
    P.isSorted = true
    return P
end

###################################################
############# Summation functions: ################
###################################################

function sumTwoSegments(s1::Segment,s2::Segment)
    vers = [s1.p1+s2.p1,
        s1.p1+s2.p2,
        s1.p2+s2.p1,
        s1.p2+s2.p2        
    ]
    poly = Polygon(vers)
    sortPolygon!(poly)
    return poly
end    

Base.:+(s1::Segment,s2::Segment) = sumTwoSegments(s1,s2)

function minkowskiSum(v::Vertex,P::Polygon)
    # this function adds v to every vertex of P
    n=length(P.vertices)
    poly=P #initial value
    for i=1:n
        poly.vertices[i] +=v
    end
    sortPolygon!(poly)
    return poly
end

function minkowskiSum(P::Polygon,v::Vertex)
    # this function adds v to every vertex of P
    n=length(P.vertices)
    poly=P #initial value
    for i=1:n
        poly.vertices[i] +=v
    end
    sortPolygon!(poly)
    return poly
end
    
Base.:(+)(v::Vertex,P::Polygon) = minkowskiSum(v,P)
Base.:(+)(P::Polygon,v::Vertex) = minkowskiSum(P,v)

"""
    minkowskiSum(P::Polygon, Q::Polygon)

Compute the Minkowski sum of two convex polygons `P` and `Q`. 

The Minkowski sum of two sets `A` and `B` in Euclidean space is the set of all points obtained by adding each point in `A` to each point in `B`. This function assumes that both `P` and `Q` are convex polygons represented by their vertices, ordered counter-clockwise starting from the vertex with the smallest y-coordinate (and the smallest x-coordinate in case of a tie).

# Arguments
- `P::Polygon`: The first convex polygon.
- `Q::Polygon`: The second convex polygon.

# Returns
- `Polygon`: A new Polygon representing the Minkowski sum of `P` and `Q`.

# Details
- The function handles different cases:
    1. Both `P` and `Q` are single vertices (points).
    2. `P` is a single vertex and `Q` is not.
    3. `Q` is a single vertex and `P` is not.
    4. Both `P` and `Q` have more than one vertex.
- In the case where both `P` and `Q` have more than one vertex, the function ensures they are sorted correctly before computing the Minkowski sum.
- The function uses the angles of the vertices in the polygons to compute the sum.
"""
function minkowskiSum(P::Polygon,Q::Polygon)
    # Computes the minkowski sum of two convex polygons: P and Q. The polygons
    # are represented by their vertices and are ordered counter clockwise such
    #* that the first vertex will be the one who has the smallest Y coordinate
    # (and smallest X coordinate in case of a tie).  This assumption is maintained
    # in twoDproj by conditions in BLPcalculator.
    
    m = length(P.vertices)
    n = length(Q.vertices)
    
# case 1: Both P and Q are length 1 (vertices)
    if m==1 && n==1 
        R = Polygon([P.vertices[1]+Q.vertices[1]])

# case 2: P is length 1 (a vertex) and Q is not    
    elseif m==1    
        R = P.vertices[1] + Q

# case 3: Q is length 1 (a vertex) and P is not
    elseif n==1
        R = Q.vertices[1] + P
    
# case 4: both Q and P have more than 1 vertex
    else
        sortPolygon!(P)
        sortPolygon!(Q)
        angP=[angles(P); angles(P)[1]] # 100 is just a big number that we know is larger
        angQ=[angles(Q); angles(Q)[1]] # than all the angles which are between 0 and 2π
    
        #m = length(P.vertices)
        #n = length(Q.vertices)
    
        PP = [P.vertices; P.vertices[1]]
        QQ = [Q.vertices; Q.vertices[1]]
    
        #println("m=",m," n=",n)
    
        #println("angP= ", angP)
        #println("angQ= ", angQ)
    
        i=1; j=1;
        #println("----- begin ----------")
        
        tol = 10^-6

        R =Polygon([])
        #R =Polygon([PP[1]+QQ[1]]) # a polygon with the sum of the two lower points as the first vertex.
        #println("R vertices: ",R.vertices)
        while (i<m+1 || j<n+1)
            R.vertices = [ R.vertices; PP[i]+QQ[j]]
            if j == n+1 #angP[i]<angQ[j] 
                #println("angP[i] is minimal")
                i +=1
            elseif i == m+1 #angQ[j]<angP[i]
                #println("angQ[j] is minimal")
                j +=1
            else
                dif = angP[i]-angQ[j]
                if dif ≤ tol
                    i +=1
                end
                if dif ≥ -tol
                    j +=1
                end
            end
            #println(i,j)
            #println("R vertices: ",R.vertices)
        end
    end
    sortPolygon!(R)
    return R
end

Base.:(+)(P::Polygon,Q::Polygon) = minkowskiSum(P,Q)

"""
    minkowskiSum(P::Polygon, s::Segment)

Compute the Minkowski sum of a convex polygon `P` and a line segment `s`. 

The function first converts the segment `s` into a two-vertex polygon `Q`. It sorts the vertices of `Q` to maintain the counter-clockwise order starting from the vertex with the smallest y-coordinate. Then, it computes the Minkowski sum of the polygon `P` and this two-vertex polygon `Q`.

# Arguments
- `P::Polygon`: The convex polygon.
- `s::Segment`: The line segment to be added to the polygon.

# Returns
- `Polygon`: A new Polygon representing the Minkowski sum of `P` and the polygon version of segment `s`.

# Details
- The function assumes that `Polygon` and `Segment` are defined types, where `Segment` consists of two endpoints `p1` and `p2`.
- The segment `s` is first transformed into a polygon `Q` with two vertices, then sorted if necessary.
- The Minkowski sum `P + Q` is then computed, where `Q` is the polygon representation of the segment `s`.
"""
function minkowskiSum(P::Polygon,s::Segment)
    # first, convert the segment to a sorted polygon with two vertices
    Q = Polygon([s.p1, s.p2])
    sortPolygon!(Q)
    
    # second, sum the two polygons
    return P+Q
end

Base.:(+)(P::Polygon,s::Segment) = minkowskiSum(P,s)


function lambdaPolygon(P::Polygon,λ::Real)
    n = length(P.vertices)
    R=P
    for i=1:n
       R.vertices[i]=λ*P.vertices[i] 
    end
    return R
end

#question: can this function be written using a map() function?

"""
    dirHausdorff(P::Polygon, Q::Polygon)

Calculate the directed Hausdorff distance from polygon `P` to polygon `Q`.

The directed Hausdorff distance is the maximum distance from any point in `P` to the closest point in `Q`. For each vertex in `P`, the function computes the minimum distance to any edge of `Q`, and then returns the maximum of these minimum distances.

# Arguments
- `P::Polygon`: The first polygon.
- `Q::Polygon`: The second polygon.

# Returns
- `Real`: The directed Hausdorff distance from `P` to `Q`.

# Details
- The function assumes that `Polygon` is a defined type with a property `vertices`, where each vertex is a point in 2D space.
- For each vertex in `P`, the function considers all edges of `Q` (formed by consecutive vertices, with the last vertex connected back to the first) and calculates the minimum distance to these edges.
- The directed Hausdorff distance is the maximum of these minimum distances.
"""
function dirHausdorff(P::Polygon,Q::Polygon)
    n = length(P.vertices)
    m = length(Q.vertices)
    dist = - Inf
    for i=1:n
        d = Inf
        for j=1:m
            next_j = j+1>m ? 1 : j+1
            d = min(d, dotDist(P.vertices[i],Segment(Q.vertices[j],Q.vertices[next_j])))
        end
        dist = max(dist,d)
    end
    return dist
end

"""
    hausdorff(P::Polygon, Q::Polygon)

Calculate the Hausdorff distance between two polygons `P` and `Q`.

The Hausdorff distance is a measure of the discrepancy between two sets. It is defined as the maximum of two directed Hausdorff distances: one from `P` to `Q` and the other from `Q` to `P`. This distance reflects the greatest of all distances from a point in one set to the closest point in the other set.

# Arguments
- `P::Polygon`: The first polygon.
- `Q::Polygon`: The second polygon.

# Returns
- `Real`: The Hausdorff distance between `P` and `Q`.

# Details
- The function computes the directed Hausdorff distance from `P` to `Q` and from `Q` to `P`, then returns the larger of these two values.
- This measure is symmetrical and captures the notion of the distance between two geometric shapes.
"""
function hausdorff(P::Polygon,Q::Polygon)
    d1 = dirHausdorff(P,Q)
    d2 = dirHausdorff(Q,P)
    return max(d1,d2)
end

#end #of module Polygons 
