import numpy as np
import matplotlib.pyplot as plt
from numpy import sin, cos, tan, pi, matmul, dot, arctan2, cross
from numpy.linalg import norm
import copy


def vec2(x, y):
    return np.array([x, y])


def vec3(x, y, z):
    return np.array([x, y, z])


def vec4(x, y, z, w):
    return np.array([x, y, z, w])


def hom4_to_can3(v):
    return vec3(v[0], v[1], v[2]) / v[3]


def hom3_to_can2(v):
    return vec2(v[0], v[1]) / v[2]


def color(r, g, b):
    return np.array([r, g, b], dtype=float)


class Line:
    def __init__(self, p, d):
        self.origin = p
        self.direction = d

    def at(self, t):
        return self.origin + self.direction * t


class Model:
    def __init__(self, verts, tris):
        self.verts = verts
        self.tris = tris


class Instance:
    def __init__(self, model, transform):
        self.model = model
        self.transform = transform
        self.verts = []
        self.tris = self.model.tris
        #for V in self.model.verts:
        #    self.verts.append(self.transform.apply_transform(V))
        self.apply_3d_transform()
        self.bounding_sphere = None
        self.set_bounding_sphere()

    def set_bounding_sphere(self):
        farthest = 0
        for V in self.verts:
            dist_from_orig = norm(V[:3]-self.transform.translation)
            if norm(V) > farthest:
                farthest = dist_from_orig
        self.bounding_sphere = Sphere(farthest, vec4(self.transform.translation[0], self.transform.translation[1], self.transform.translation[2], 1))

    def transform_3d(self, transform):
        self.transform = transform
        self.apply_3d_transform()
        self.set_bounding_sphere()
    def bounding_sphere_to_camera(self, cam):
        self.bounding_sphere.center = matmul(cam.M_cam, self.bounding_sphere.center)

    def apply_3d_transform(self):
        M3d = self.transform.M_trf
        transformed_verts = []
        for V in self.model.verts:
            transformed_verts.append(matmul(M3d, V))
        self.verts = transformed_verts

    def apply_cam_transform(self, cam):
        Mcam = cam.M_cam
        transformed_verts = []
        for V in self.verts:
            transformed_verts.append(matmul(Mcam, V))
        self.verts = transformed_verts
        self.bounding_sphere_to_camera(cam)

class Triangle:
    def __init__(self, v0, v1, v2, data):
        self.vert = [v0, v1, v2]
        self.data = data


class Sphere:
    def __init__(self, r, C):
        self.radius = r
        self.center = C


class material:
    def __init__(self, color, roughness, metallic, normal):
        self.color = color
        self.roughness = roughness
        self.metallic = metallic
        self.normal = normal


class TriangleData:
    def __init__(self, z, u, v, material):
        self.z = z
        self.u = u
        self.v = v
        self.material = material


class Plane:
    def __init__(self, n, d):
        self.normal = n
        self.D = d


class Transform:
    def __init__(self, scale, rotation, translation):
        self.scale = scale
        self.rotation = rotation
        self.translation = translation
        self.M_trf = None
        self.M_s = None
        self.M_r = None
        self.M_t = None
        self.inv_M_s = None
        self.inv_M_r = None
        self.inv_M_t = None
        self.set_transform_matrix()

    def set_transform_matrix(self):
        a, b, c = degree_to_radians(self.rotation)
        self.M_s = vec4(vec4(self.scale[0], 0, 0, 0),
                        vec4(0, self.scale[1], 0, 0),
                        vec4(0, 0, self.scale[2], 0),
                        vec4(0, 0, 0, 1))
        self.M_r = vec4(vec4(cos(b) * cos(c), sin(a) * sin(b) * cos(c) - cos(a) * sin(c),
                             cos(a) * sin(b) * cos(c) + sin(a) * sin(c), 0),
                        vec4(cos(b) * sin(c), sin(a) * sin(b) * sin(c) + cos(a) * cos(c),
                             cos(a) * sin(b) * sin(c) - sin(a) * cos(c), 0),
                        vec4(-sin(b), sin(a) * cos(b), cos(a) * cos(b), 0),
                        vec4(0, 0, 0, 1))
        self.M_t = vec4(vec4(1, 0, 0, self.translation[0]),
                        vec4(0, 1, 0, self.translation[1]),
                        vec4(0, 0, 1, self.translation[2]),
                        vec4(0, 0, 0, 1))
        self.M_trf = np.matmul(np.matmul(self.M_t, self.M_r), self.M_s)

    def apply_transform(self, V):
        return matmul(self.M_trf, V)

class Camera:
    def __init__(self, d, transform, res, FoV, AR):
        self.d = d
        self.transform = transform
        self.HFoV = degree_to_radians(FoV/2)
        self.AspectRatio = AR
        self.res = res
        self.Vw = 2*self.d * tan(self.HFoV)
        self.Vh = self.Vw / self.AspectRatio
        self.VFoV = arctan2(self.Vh, 2*self.d)
        self.Cw = self.res
        self.Ch = int(self.Cw / self.AspectRatio)
        self.img = np.ones((self.Cw, self.Ch, 3), dtype=float)
        self.depth_buffer = np.ones((self.Cw, self.Ch), dtype=float) * np.infty
        self.M_cam = None
        self.M_proj = None
        self.set_camera_matrix()
        self.set_projection_matrix()
        self.clip_planes = []
        self.set_clip_planes()

    def putPixel(self, x, y, c):
        Cx = x + self.Cw // 2
        Cy = -y + self.Ch // 2
        if (Cx >= self.Cw) | (Cx < 0): return
        if (Cy >= self.Ch) | (Cy < 0): return
        self.img[Cx, Cy, :] = c

    def set_camera_matrix(self):
        self.M_cam = np.matmul(np.linalg.inv(self.transform.M_r), np.linalg.inv(self.transform.M_t))

    def set_projection_matrix(self):
        self.M_proj = vec3(vec4(self.d * self.Cw / self.Vw, 0, 0, 0),
                           vec4(0, self.d * self.Ch / self.Vh, 0, 0),
                           vec4(0, 0, 1, 0))

    def set_clip_planes(self):
        # near
        self.clip_planes.append(Plane(vec4( 0,  0, 1, 0), -self.d))
        # top
        self.clip_planes.append(Plane(vec4( 0, -cos(self.VFoV), sin(self.VFoV), 0), 0))
        # bottom
        self.clip_planes.append(Plane(vec4( 0,  cos(self.VFoV), sin(self.VFoV), 0), 0))
        # left
        self.clip_planes.append(Plane(vec4( cos(self.HFoV),  0, sin(self.HFoV), 0), 0))
        # right
        self.clip_planes.append(Plane(vec4(-cos(self.HFoV),  0, sin(self.HFoV), 0), 0))


def normalise(V):
    return V/norm(V)

def interpolate(i0, d0, i1, d1):
    if i0 == i1:
        return np.array([d0])
    values = np.zeros(i1 - i0 + 1)
    a = (d1 - d0) / (i1 - i0)
    d = d0
    for i in range(i1 - i0 + 1):
        values[i] = d
        d = d + a
    return values


def swap3(P0, P1):
    x1 = P1[0]
    y1 = P1[1]
    z1 = P1[2]
    x0 = P0[0]
    y0 = P0[1]
    z0 = P0[2]
    return x0, y0, z0, x1, y1, z1


def swap2(P0, P1):
    x1 = P1[0]
    y1 = P1[1]
    x0 = P0[0]
    y0 = P0[1]
    return x0, y0, x1, y1


def DrawLine(P0, P1, color, cam):
    x0, y0, x1, y1 = swap2(P0, P1)
    if abs(P1[0] - P0[0]) > abs(P1[1] - P0[1]):
        # horizontal
        if P0[0] > P1[0]:
            x0, y0, x1, y1 = swap2(P1, P0)
        ys = interpolate(x0, y0, x1, y1)
        for x in range(x0, x1 + 1):
            cam.putPixel(x, int(ys[x - x0]), color)
    else:
        # vertical
        if P0[1] > P1[1]:
            x0, y0, x1, y1 = swap2(P1, P0)
        xs = interpolate(y0, x0, y1, x1)
        for y in range(y0, y1 + 1):
            cam.putPixel(int(xs[y - y0]), y, color)


def DrawWireframeTriangle(P0, P1, P2, color, cam):
    DrawLine(P0, P1, color, cam)
    DrawLine(P1, P2, color, cam)
    DrawLine(P2, P0, color, cam)


def DrawTriangle(P0, P1, P2, h, color, cam):
    x0 = P0[0]/P0[2]
    y0 = P0[1]/P0[2]
    x1 = P1[0]/P1[2]
    y1 = P1[1]/P1[2]
    x2 = P2[0]/P2[2]
    y2 = P2[1]/P2[2]
    z0 = 1/P0[2]
    z1 = 1/P1[2]
    z2 = 1/P2[2]
    if y1 < y0: x0, y0, z0, x1, y1, z1 = swap3(vec3(x1, y1, z1), vec3(x0, y0, z0))
    if y2 < y0: x0, y0, z0, x2, y2, z2 = swap3(vec3(x2, y2, z2), vec3(x0, y0, z0))
    if y2 < y1: x1, y1, z1, x2, y2, z2 = swap3(vec3(x2, y2, z2), vec3(x1, y1, z1))
    y0 = int(y0)
    y1 = int(y1)
    y2 = int(y2)
    x01 = interpolate(y0, x0, y1, x1)
    x12 = interpolate(y1, x1, y2, x2)
    x02 = interpolate(y0, x0, y2, x2)
    x012 = np.concatenate((x01[0:-1], x12))
    z01 = interpolate(y0, z0, y1, z1)
    z12 = interpolate(y1, z1, y2, z2)
    z02 = interpolate(y0, z0, y2, z2)
    z012 = np.concatenate((z01[0:-1], z12))
    if x012[len(x02) // 2] < x02[len(x02) // 2]:
        xleft = x012
        xright = x02
        zleft = z012
        zright = z02
    else:
        xleft = x02
        xright = x012
        zleft = z02
        zright = z012
    for y in range(y0, y2 + 1):
        xl = int(xleft[y - y0])
        xr = int(xright[y - y0])
        z_segment = interpolate(xl, zleft[y - y0], xr, zright[y - y0]) / 1
        for x in range(xl, xr):
            z = z_segment[x - xl]
            if 1/z < cam.depth_buffer[x, y]:
                cam.depth_buffer[x, y] = 1/z
                cam.putPixel(x, y, color)


def DrawTriangleOld(P0, P1, P2, color, cam):
    x0 = P0[0]
    y0 = P0[1]
    x1 = P1[0]
    y1 = P1[1]
    x2 = P2[0]
    y2 = P2[1]
    if y1 < y0: x0, y0, x1, y1 = swap2(P1, P0)
    if y2 < y0: x0, y0, x2, y2 = swap2(P2, P0)
    if y2 < y1: x1, y1, x2, y2 = swap2(P2, P1)
    x01 = interpolate(y0, x0, y1, x1)
    x12 = interpolate(y1, x1, y2, x2)
    x02 = interpolate(y0, x0, y2, x2)
    x012 = np.concatenate((x01[0:-1], x12))
    if x012[len(x02) // 2] < x02[len(x02) // 2]:
        xleft = x012
        xright = x02
    else:
        xleft = x02
        xright = x012
    for y in range(y0, y2 + 1):
        xl = int(xleft[y - y0])
        xr = int(xright[y - y0])
        for x in range(xl, xr):
            cam.putPixel(x, y, color)

def degree_to_radians(theta):
    return pi * theta / 180


def scale(v, s):
    return v * s[0]


def rotate(v, r):
    r = degree_to_radians(r[1])
    Mr = vec4(vec4(cos(r), 0, sin(r), 0), vec4(0, 1, 0, 0), vec4(-sin(r), 0, cos(r), 0), 0)
    return np.matmul(Mr, v)


def translate(v, t):
    return v + vec4(t[0], t[1], t[2], 0)


def ApplyTransform_old(v, transform):
    scaled = scale(v, transform.scale)
    rotated = rotate(scaled, transform.rotation)
    translated = translate(rotated, transform.translation)
    return translated


def ApplyTransform(v, transform):
    return np.matmul(transform.M_trf, v)


def ViewportToCanvas(x, y, cam):
    return [int(x * cam.Cw / cam.Vw), int(y * cam.Ch / cam.Vh)]


def ProjectVertex_old(v, cam):
    return ViewportToCanvas(v[0] * cam.d / v[2], v[1] * cam.d / v[2], cam)


def ProjectVertex(v, cam):
    M = cam.M_proj
    N = np.matmul(cam.M_proj, v).astype(int)
    return hom3_to_can2(np.matmul(cam.M_proj, v)).astype(int)


#def ProjectVertex(v, M_trf, cam):
#    M = cam.M_proj
#    N = np.matmul(cam.M_proj, v).astype(int)
#    return hom3_to_can2(np.matmul(cam.M_proj, v)).astype(int)


def RenderTriangle(triangle, projected, cam):
    # h = [triangle.z, triangle.u, triangle.vert]
    DrawTriangle(projected[triangle.vert[0]],
                 projected[triangle.vert[1]],
                 projected[triangle.vert[2]],
                 0, triangle.data, cam)


def RenderModel(model, cam):
    projected = []
    for V in model.verts:
        projected.append(ProjectVertex(V, cam))
    for T in model.tris:
        RenderTriangle(T, projected, cam)


def RenderInstance(instance, cam):
    projected = []
    # model = instance.model
    # Mm = instance.transform.M_trf
    # Mc = cam.M_cam
    Mp = cam.M_proj
    # F = np.matmul(np.matmul(Mp, Mc), Mm)
    # F = matmul(Mp, Mc)
    for V in instance.verts:
        projected.append(np.matmul(Mp, V))
        # Vtrans = ApplyTransform(V, instance.transform)
        # projected.append(ProjectVertex(Vtrans, cam))
    for T in instance.tris:
        RenderTriangle(T, projected, cam)


def SignedDistance(plane, point):
    return point[0] * plane.normal[0] + point[1] * plane.normal[1] + point[2] * plane.normal[2] + plane.D


def ClipInstance(instance, cam):
    temp_instance = instance
    for P in cam.clip_planes:
        d = SignedDistance(P, temp_instance.bounding_sphere.center)
        if d < -temp_instance.bounding_sphere.radius:
            return None
        elif d < temp_instance.bounding_sphere.radius:
            clipped_instance = copy.deepcopy(temp_instance)
            clipped_instance.tris, clipped_instance.verts = ClipTriangles(P, temp_instance.tris, temp_instance.verts)
            temp_instance = clipped_instance
    return temp_instance

def Intersect(plane, line):
    D = plane.D
    n = plane.normal[:3]
    p = line.origin[:3]
    d = line.direction[:3]
    t = (D - dot(n, p))/dot(n, d)
    return line.at(t)

def ClipTriangles(plane, tris, verts):
    clipped_triangles = []
    clipped_verts = copy.deepcopy(verts)
    added_verts = []
    vert_count = len(verts)
    for triangle in tris:
        d0 = SignedDistance(plane, verts[triangle.vert[0]])
        d1 = SignedDistance(plane, verts[triangle.vert[1]])
        d2 = SignedDistance(plane, verts[triangle.vert[2]])
        d = vec3(d0, d1, d2)
        idx = d.argsort()
        V_hi = verts[triangle.vert[idx[2]]]
        V_md = verts[triangle.vert[idx[1]]]
        V_lo = verts[triangle.vert[idx[0]]]
        if d[idx[0]] > -0.001:
            clipped_triangles.append(triangle)
        elif d[idx[1]] > -0.001:
            I1 = Intersect(plane, Line(V_hi, V_hi-V_lo))
            I2 = Intersect(plane, Line(V_md, V_md-V_lo))
            added_verts.append(I1)
            I1idx = vert_count + len(added_verts) - 1
            added_verts.append(I2)
            I2idx = vert_count + len(added_verts) - 1
            clipped_triangles.append(Triangle(triangle.vert[idx[2]], triangle.vert[idx[1]], I1idx, triangle.color))
            clipped_triangles.append(Triangle(triangle.vert[idx[1]], I1idx, I2idx, triangle.color))

        elif d[idx[2]] > -0.001:
            I1 = Intersect(plane, Line(V_hi, V_hi-V_lo))
            I2 = Intersect(plane, Line(V_hi, V_hi-V_md))
            added_verts.append(I1)
            I1idx = vert_count + len(added_verts) - 1
            added_verts.append(I2)
            I2idx = vert_count + len(added_verts) - 1
            clipped_triangles.append(Triangle(triangle.vert[idx[2]], I1idx, I2idx, triangle.color))
    for V in added_verts:
        clipped_verts.append(V)
    return clipped_triangles, clipped_verts

def ClipScene(scene, cam):
    clipped_instances = []
    for I in scene:
        I.apply_cam_transform(cam)
        clipped_instance = ClipInstance(I, cam)
        if clipped_instance is not None:
            clipped_instances.append(clipped_instance)
    return clipped_instances


def BackFaceCulling(instance):
    culled_tris = []
    for triangle in instance.tris:
        V1 = instance.verts[triangle.vert[1]] - instance.verts[triangle.vert[0]]
        V2 = instance.verts[triangle.vert[2]] - instance.verts[triangle.vert[0]]
        N = cross(V1[:3], V2[:3])
        if dot(N, instance.verts[triangle.vert[0]][:3]) < 0:
            culled_tris.append(triangle)
    instance.tris = culled_tris

def RenderScene(scene, cam):
    clipped_scene = ClipScene(scene, cam)
    # verts2 = [vec4(-1, -1, 1, 1),
    #           vec4(-2, -1, 2, 1),
    #           vec4(-2, 1, 2, 1),
    #           vec4(-1, 1, 1, 1)]
    # tris2 = [Triangle(0, 1, 2, black), Triangle(1, 2, 3, black)]
    # plane = Model(verts2, tris2)
    # clipped_scene.append(Instance(plane, Transform(vec3(1, 1, 1), vec3(0, 0, 0), vec3(0, 0, 0))))
    for I in clipped_scene:
        BackFaceCulling(I)
        RenderInstance(I, cam)


black = BLACK = color(0, 0, 0)
white = WHITE = color(1, 1, 1)
red = RED = color(1, 0, 0)
green = GREEN = color(0, 1, 0)
blue = BLUE = color(0, 0, 1)
yellow = YELLOW = color(1, 1, 0)
purple = MAGENTA = color(1, 0, 1)
cyan = CYAN = color(0, 1, 1)


def render():
    verts = [vec4(1, 1, 1, 1),
             vec4(-1, 1, 1, 1),
             vec4(-1, -1, 1, 1),
             vec4(1, -1, 1, 1),
             vec4(1, 1, -1, 1),
             vec4(-1, 1, -1, 1),
             vec4(-1, -1, -1, 1),
             vec4(1, -1, -1, 1)]
    tris = [Triangle(0, 1, 2, red),
            Triangle(0, 2, 3, red),
            Triangle(4, 0, 3, green),
            Triangle(4, 3, 7, green),
            Triangle(5, 4, 7, blue),
            Triangle(5, 7, 6, blue),
            Triangle(1, 5, 6, yellow),
            Triangle(1, 6, 2, yellow),
            Triangle(4, 5, 1, purple),
            Triangle(4, 1, 0, purple),
            Triangle(2, 6, 7, cyan),
            Triangle(2, 7, 3, cyan)]
    scene = []
    cube = Model(verts, tris)
    scene.append(Instance(cube, Transform(vec3(1, 1, 1), vec3(0, -45, 0), vec3(-1.5, -.5, 5.5))))
    scene.append(Instance(cube, Transform(vec3(1, 1, 1), vec3(0, 135, 0), vec3(1.25, 3, 6.5))))
    cam = Camera(1, Transform(vec3(1, 1, 1), vec3(0, 0, 0), vec3(0, 1, 0)), 1200, 80, 1.2)
    RenderScene(scene, cam)
    plt.imshow(cam.img.transpose((1, 0, 2)), origin='upper')
    plt.show()


render()