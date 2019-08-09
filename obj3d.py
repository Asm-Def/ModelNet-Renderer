import cupy as cp
import numpy as np
import math
import glfw
from OpenGL import GL
from OpenGL.GL import shaders
import cv2 as cv
from ctypes import c_void_p
calc_face_norm = cp.RawKernel(r'''
    extern "C" __global__
    void calc_face_norm(const float* vertices, const int* face, float* face_norm) {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        int a = face[tid*3], b = face[tid*3+1], c = face[tid*3+2];
        float3 v1 = make_float3(vertices[a*3], vertices[a*3+1], vertices[a*3+2]);
        float3 v2 = make_float3(vertices[b*3], vertices[b*3+1], vertices[b*3+2]);
        float3 v3 = make_float3(vertices[c*3], vertices[c*3+1], vertices[c*3+2]);
        float3 l1 = make_float3(v2.x-v1.x, v2.y-v1.y, v2.z-v1.z);
        float3 l2 = make_float3(v3.x-v1.x, v3.y-v1.y, v3.z-v1.z);
        float3 ans = make_float3(l1.y*l2.z-l2.y*l1.z, l1.z*l2.x-l2.z*l1.x, l1.x*l2.y-l2.x*l1.y);
        face_norm[tid*3] = ans.x;
        face_norm[tid*3+1] = ans.y;
        face_norm[tid*3+2] = ans.z;
    }
''', 'calc_face_norm')

calc_vertex_norm = cp.RawKernel(r'''
    extern "C" __global__
    void calc_vertex_norm(const float* face_norm, const int* face, float* vert_norm, float* vertex_weight) {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        int a = face[tid*3], b = face[tid*3+1], c = face[tid*3+2];
        float xx = face_norm[tid*3], yy = face_norm[tid*3+1], zz = face_norm[tid*3+2];
        vert_norm[a*3] += xx;
        vert_norm[a*3+1] += yy;
        vert_norm[a*3+2] += zz;
        vert_norm[b*3] += xx;
        vert_norm[b*3+1] += yy;
        vert_norm[b*3+2] += zz;
        vert_norm[c*3] += xx;
        vert_norm[c*3+1] += yy;
        vert_norm[c*3+2] += zz;
        float area = sqrt(xx*xx + yy*yy + zz*zz) / 3;
        vertex_weight[a] += area;
        vertex_weight[b] += area;
        vertex_weight[c] += area;
}
''', 'calc_vertex_norm')

calc_avg_vertex = cp.ReductionKernel(
    'T x, T w',
    'T y',
    'x * w',
    'a + b',
    'y = a',
    '0',
    'calc_avg_vertex'
)

class Obj3D(object):
    def __init__(self, filename, lines=None):
        if lines is None:
            file = open(filename, 'r')
            lines = file.readlines()
        iter = 0

        if lines[0].rstrip() == 'OFF':
            self.numVertices, self.numFaces, self.numEdges = (int(x) for x in str.split(lines[1]))
            iter = 2
        elif lines[0][0:3] == 'OFF':
            self.numVertices, self.numFaces, self.numEdges = (int(x) for x in str.split(lines[0][3:]))
            iter = 1

        print('building arrays')
        self.vertices = cp.empty((self.numVertices, 3), dtype='float32')
        self.faces = cp.empty((self.numFaces, 3), dtype='int32')
        self.time_value = 0.0
        self.theta = math.radians(30)
        self.face_norm = cp.empty((self.numFaces, 3), dtype='float32')
        self.ver_norm = cp.zeros((self.numVertices, 3), dtype='float32')

        self.use_ver_norm = True
        if self.use_ver_norm:
            self.data = cp.empty((self.numVertices, 6), dtype='float32')
        else:
            self.data = cp.empty((3 * self.numFaces, 6), dtype='float32')

        print('reading file ' + filename)
        try:
            for i in range(self.numVertices):
                tmp = [float(x) for x in str.split(lines[iter+i])[0:3]]
                self.vertices[i] = cp.array([tmp[0], tmp[2], tmp[1]])

            vertex_avg = cp.zeros((3,), dtype='float64')

            iter += self.numVertices

            for i in range(self.numFaces):
                cur_ind = [int(x) for x in lines[iter+i].split()[1:4]]
                self.faces[i] = cp.array(cur_ind, dtype='int32')

            print('done reading')

            self.vertex_weight = cp.zeros((self.numVertices,), dtype='float32')
            calc_face_norm((self.numFaces,), (1,), (self.vertices, self.faces, self.face_norm))
            calc_vertex_norm((self.numFaces,), (1,), (self.face_norm, self.faces, self.ver_norm, self.vertex_weight))
            vertex_avg = calc_avg_vertex(self.vertices, self.vertex_weight.reshape(self.numVertices, 1), axis=0)
            vertex_avg /= self.vertex_weight.sum()
            self.vertices -= vertex_avg
            mx = max(self.vertices.max(), -self.vertices.min())
            self.vertices = self.vertices * 0.7 / mx
            print('done calculating')

            if self.use_ver_norm:
                self.data = cp.concatenate((self.vertices, self.ver_norm), 1)
            else:
                for i in range(self.numFaces):
                    norm = self.face_norm[i]
                    vert = self.faces[i]
                    for j in range(3):
                        self.data[i*3+j] = cp.concatenate((self.vertices[vert[j]], norm))

        except Exception as e:
            print('Wrong Format', e)
            raise Exception('Wrong Format')

        print('init_gl')
        self.init_gl()
        print('done')

    def __del__(self):
        GL.glDeleteProgram(self.program)
        GL.glDeleteBuffers(1, self.VBO)
        GL.glDeleteVertexArrays(1, self.VAO)

    def init_gl(self):
        self.VAO = GL.glGenVertexArrays(1)
        self.VBO = GL.glGenBuffers(1)
        self.EBO = GL.glGenBuffers(1)

        vertex_shader_source = open('vertex.glsl', 'r').read()
        frag_shader_source = open('frag.glsl', 'r').read()
        self.program = shaders.compileProgram(
            shaders.compileShader(vertex_shader_source, GL.GL_VERTEX_SHADER),
            shaders.compileShader(frag_shader_source, GL.GL_FRAGMENT_SHADER)
        )

        GL.glBindVertexArray(self.VAO)

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.VBO)

        GL.glBufferData(GL.GL_ARRAY_BUFFER, self.data.flatten().get(), GL.GL_STATIC_DRAW)
        #GL.glBufferData(GL.GL_ARRAY_BUFFER, np.concatenate((self.vertices, self.ver_norm), 1), GL.GL_STATIC_DRAW)

        if self.use_ver_norm:
            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.EBO)
            GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, self.faces.flatten().get(), GL.GL_STATIC_DRAW)

        a_pos = GL.glGetAttribLocation(self.program, 'position')
        a_normal = GL.glGetAttribLocation(self.program, 'normal')
        # print('a_pos,anorm =', a_pos, a_normal)
        GL.glEnableVertexAttribArray(a_pos)
        GL.glEnableVertexAttribArray(a_normal)
        GL.glVertexAttribPointer(a_pos, 3, GL.GL_FLOAT, False, 6 * 4, c_void_p(0))
        GL.glVertexAttribPointer(a_normal, 3, GL.GL_FLOAT, False, 6 * 4, c_void_p(3 * 4))

        GL.glBindVertexArray(0)


    def draw(self):
        GL.glClearColor(1.0, 1.0, 1.0, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glBindVertexArray(self.VAO)
        #time_value = glfw.get_time() * 0.7
        time_value = self.time_value
        #theta = math.radians(30)
        theta = self.theta

        rotate = np.array([
            [math.cos(time_value), 0., -math.sin(time_value), 0.],
            [0., 1., 0., 0.],
            [math.sin(time_value), 0., math.cos(time_value), 0.],
            [0., 0., 0., 1.]
        ], dtype='float32')

        view = np.array([
            [1, 0, 0, 0],
            # [0, 1, 0, 0],
            # [0, 0, 1, 0],
            [0, math.cos(theta), -math.sin(theta), 0],
            [0, math.sin(theta), math.cos(theta), 0],
            [0, 0, 0, 1]
        ], dtype='float32')

        projection = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype='float32')

        model_mat = rotate

        model_location = GL.glGetUniformLocation(self.program, 'model')
        view_location = GL.glGetUniformLocation(self.program, 'view')
        proj_location = GL.glGetUniformLocation(self.program, 'projection')

        GL.glUseProgram(self.program)
        GL.glUniformMatrix4fv(model_location, 1, False, model_mat)
        GL.glUniformMatrix4fv(view_location, 1, False, view)
        GL.glUniformMatrix4fv(proj_location, 1, False, projection)

        if self.use_ver_norm:
            GL.glDrawElements(GL.GL_TRIANGLES, self.numFaces * 3, GL.GL_UNSIGNED_INT, c_void_p(0))
        else:
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, self.numFaces * 3)

        GL.glBindVertexArray(0)
