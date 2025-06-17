import matplotlib.pyplot as plt
import numpy as np
from numpy import inf, sqrt, dot
from numpy.linalg import norm

Cw = 200
Ch = 200
Vw = 1
Vh = 1


im = np.zeros([Cw, Ch, 3], dtype=int)
O = np.array([0, 0, 0])
d = 1

background_color = np.array([0.5, 0.6, 0.7])


class Sphere:
    def __init__(self, r, c, col, s, rfl):
        self.radius = r
        self.center = np.array(c)
        self.color = np.array(col)
        self.specular = s
        self.reflective = rfl


class Light:
    def __init__(self, tp, ints, dir=None, pos=None):
        self.type = tp
        self.direction = np.array(dir)
        self.position = np.array(pos)
        self.intensity = ints


class Scene:
    def __init__(self, s, l):
        self.spheres = s
        self.lights = l


spheres = [Sphere(1, [0, -1, 3], [255, 0, 0], 500, 0.2),
           Sphere(1, [2, 0, 4], [0, 0, 255], 500, 0.3),
           Sphere(1, [-2, 0, 4], [0, 255, 0], 10, 0.4),
           Sphere(5000, [0, -5001, 0], [255, 255, 0], 1000, 0.5)]

lights = [Light('ambient', 0.2),
          Light('point', 0.6, pos=[2, 1, 0]),
          Light('directional', 0.2, dir=[1, 4, 4])]

scene = Scene(spheres, lights)

def canvasToViewport(x, y):
    return np.array([(x-Cw/2)*Vw/Cw, (y-Ch/2)*Vh/Ch, d])


def intersectRaySphere(O, D, sphere):
    r = sphere.radius
    CO = O - sphere.center

    a = dot(D, D)
    b = 2*dot(CO, D)
    c = dot(CO, CO) - r*r

    discriminant = b*b - 4*a*c
    if discriminant < 0:
        return inf, inf
    t1 = (-b + sqrt(discriminant))/(2*a)
    t2 = (-b - sqrt(discriminant))/(2*a)
    return t1, t2


def closestIntersection(O, D, tmin, tmax):
    closest_t = inf
    closest_sphere = None
    for sphere in scene.spheres:
        t1, t2 = intersectRaySphere(O, D, sphere)
        if tmin < t1 < tmax and t1 < closest_t:
            closest_t = t1
            closest_sphere = sphere
        if tmin < t2 < tmax and t2 < closest_t:
            closest_t = t2
            closest_sphere = sphere
    return closest_sphere, closest_t


def reflectRay(R, N):
    return 2 * N * dot(N, R) - R

def computeLighting(P, N, V, s):
    i = 0.0
    for light in scene.lights:
        if light.type == 'ambient':
            i += light.intensity
        else:
            if light.type == 'point':
                L = light.position - P
                tmax = 1
            else:
                L = light.direction
                tmax = inf

            # Shadow check
            shadow_sphere, shadow_t = closestIntersection(P, L, 0.001, tmax)
            if shadow_sphere is not None:
                break

            # Diffuse
            n_dot_l = dot(N, L)
            if n_dot_l > 0:
                i += light.intensity * n_dot_l/(norm(L) * norm(N))

            # Specular
            if s != -1:
                R = reflectRay(L, N)
                r_dot_v = dot(R, V)
                if r_dot_v > 0:
                    i += light.intensity * pow(r_dot_v/(norm(R)*norm(V)), s)
    return i


def traceRay(O, D, tmin, tmax, recursion_depth):
    closest_sphere, closest_t = closestIntersection(O, D, tmin, tmax)
    if closest_sphere is None:
        return background_color
    # Local
    P = O + closest_t * D
    N = P - closest_sphere.center
    N = N/norm(N)
    local_color = computeLighting(P, N, -D, closest_sphere.specular) * closest_sphere.color

    # Terminate recursion
    r = closest_sphere.reflective
    if recursion_depth <= 0 or r <= 0:
        return local_color

    # Compute reflexion
    R = reflectRay(-D, N)
    reflect_color = traceRay(P, R, 0.001, inf, recursion_depth-1)

    return np.clip(local_color * (1-r) + reflect_color * r, 0, 255)


for x in range(0, Cw):
    for y in range(0, Ch):
        D = canvasToViewport(x, y)
        color = traceRay(O, D, d, inf, 3)
        im[y, x] = color

plt.imshow(im, origin='lower')

plt.show()
