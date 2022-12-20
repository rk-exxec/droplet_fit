import numpy as np
import numba as nb




@nb.vectorize(["float32(float32, float32, float32, float32)", "double(double, double, double, double)"])
def get_root(r0, z0, z1, g) -> float:
    """
    Implementation of Eqn. 24 \\
    https://www.geometrictools.com/Documentation/DistancePointEllipseEllipsoid.pdf 
    
    """
    max_iter = 149 # for float or 1074 for double
    n0 = r0*z0
    s0 = z1 - 1
    s1  = ( 0 if g < 0 else np.sqrt(n0**2 + z1**2))
    s = 0
    for _ in range(0, max_iter):
        s = (s0 + s1) / 2
        if s == s0 or s == s1 : break
        ratio0 = n0/(s + r0)
        ratio1 = z1/(s + 1)
        g = ratio0**2 + ratio1**2 - 1
        if g > 0:
            s0 = s
        elif g < 0:
            s1 = s
        else:
            break
    return s

# decorator makes function accept numpy array and applies calculation to every element, returns numpy array of results
@nb.guvectorize(['void(double[:], float32, float32, double[:])'], "(n),(),()->()", target='parallel', cache=True)
def calc_residuals(point, a, b, out):
    """calculate square distance from point to nearest point on ellipse
    https://www.geometrictools.com/Documentation/DistancePointEllipseEllipsoid.pdf \\
    y0,y1 == point_x,point_y \\
    z0,z1 == point_x_scaled,point_y_scaled \\
    r0 == ratio_sq_ell_ax \\
    e0,e1 == a,b \\
    x0,x1 == ellipse_point_x,ellipse_point_y \\
    converts every calculation to 1st quadrant, should be the same, as ellipse is symmetric to both axes
    ---
    :param point: point to check
    :type point: np.ndarray
    :param a,b: ellipse half-axes
    :return: distance of point to ellipse squared
    """
    point_x,point_y = point
    # since the point is in the ellipse coord system, the problem is symmetric, 
    # thus for easier calculation, all points are treated as 1. quadrant points
    point_x = abs(point_x)
    point_y = abs(point_y)
    if(point_y > 0):
        if (point_x > 0):
            point_x_scaled = point_x/a
            point_y_scaled = point_y/b
            g = point_x_scaled**2 + point_y_scaled**2 - 1
            if (g != 0):
                ratio_sq_ell_ax = (a/b)**2
                root = get_root(ratio_sq_ell_ax, point_x_scaled, point_y_scaled, g)
                ellipse_point_x = ratio_sq_ell_ax*point_x/(root + ratio_sq_ell_ax)
                ellipse_point_y = point_y/(root + 1)
            else:
                ellipse_point_x,ellipse_point_y = point_x,point_y
        else:
            ellipse_point_x,ellipse_point_y = 0,b
    else:
        num = a*point_x
        den = a**2 - b**2
        if num < den:
            frac = num/den
            ellipse_point_x = a*frac
            ellipse_point_y = b*np.sqrt(1-frac**2)
        else:
            ellipse_point_x,ellipse_point_y = a,0
    #return squared distance
    out[0] = (point_x - ellipse_point_x)**2 + (point_y - ellipse_point_y)**2
