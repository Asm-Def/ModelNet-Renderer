import glfw
from OpenGL import GL
import numpy as np
from obj3d import *
import cv2 as cv
import glob
import os
import threading
from queue import Queue

def init():
    np.float128 = np.float64  # 解决windows下出现的问题
    np.complex256 = np.complex64

    glfw.init()
    glfw.window_hint(glfw.SAMPLES, 8)  # 反走样
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.RESIZABLE, GL.GL_FALSE)
    glfw.window_hint(glfw.OPENGL_DEBUG_CONTEXT, GL.GL_TRUE)


def main():
    init()
    filename = 'Preview'
    window = glfw.create_window(1024, 1024, filename.split('/')[-1], None, None)
    glfw.make_context_current(window)
    width, height = glfw.get_framebuffer_size(window)
    GL.glViewport(0, 0, width, height)

    def key_callback(window, key, scancode, action, mode):
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            glfw.set_window_should_close(window, True)

    glfw.set_key_callback(window, key_callback)

    GL.glEnable(GL.GL_MULTISAMPLE)
    GL.glEnable(GL.GL_DEPTH_TEST)
    GL.glEnable(GL.GL_BLEND)
    # GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)
    GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)

    draw_files(window)

    glfw.terminate()

def draw(window, filename, path, lines):
    filename = filename.replace('\\', '/')
    glfw.poll_events()
    shape = Obj3D(filename, lines)
    width, height = glfw.get_framebuffer_size(window)

    index = filename.split('/')[-1].split('.')[0]
    for V in range(12):
        while True:
            shape.draw()
            glfw.poll_events()
            GL.glReadBuffer(GL.GL_BACK)
            img = GL.glReadPixels(0, 0, width, height, GL.GL_RGB, GL.GL_UNSIGNED_BYTE)
            img = np.frombuffer(img, dtype='uint8')
            if len(img) != 1024 * 1024 * 3 or img[0] == 0:
                print("Image Error")
                glfw.terminate()
                init()
                del shape
                shape = Obj3D(filename, lines)
                shape.time_value += math.radians(30) * V

                input()
            else:
                break

        shape.time_value += math.radians(30)
        img = img.reshape((width, height, 3))[::-1, :, ::-1]
        tmp = np.empty((width//8, height//8, 3))
        for i in range(width//8):
            for j in range(height//8):
                for t in range(3):
                    tmp[i,j,t] = img[i*8:i*8+8, j*8:j*8+8, t].sum() / 64
        img = tmp
        # print(img)
        glfw.swap_buffers(window)
        out_name = path + '/' + index + '_' + str(V) + '.png'
        print(out_name)
        # cv.imshow('preview', img)
        cv.imwrite(out_name, img)


def draw_files(window):
    classnames=['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair',
                         'cone','cup','curtain','desk','door','dresser','flower_pot','glass_box',
                         'guitar','keyboard','lamp','laptop','mantel','monitor','night_stand',
                         'person','piano','plant','radio','range_hood','sink','sofa','stairs',
                         'stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox']
    train_meshes = 'G:/JasonWu/ModelNet/ModelNet40/{}/train'
    test_meshes = 'G:/JasonWu/ModelNet/ModelNet40/{}/test'
    train_img = 'data/train'
    test_img = 'data/test'

    if not os.path.exists('data'):
        os.mkdir('data')
    if not os.path.exists(train_img):
        os.mkdir(train_img)
    if not os.path.exists(test_img):
        os.mkdir(test_img)

    def read_meshes(train_files, test_files, sem, classname, train_filenames, test_filenames):
        for filename in train_filenames:
            file = open(filename, 'r')
            train_files.put(file.readlines())
            sem.release()

        for filename in test_filenames:
            file = open(filename, 'r')
            test_files.put(file.readlines())
            sem.release()

    for classname in classnames:
        train_path = train_meshes.format(classname)
        train_filenames = glob.glob(train_path + '/*.off')
        test_path = test_meshes.format(classname)
        test_filenames = glob.glob(test_path + '/*.off')

        # train_files = Queue()
        # test_files = Queue()
        # sem = threading.Semaphore(0)
        # args = (train_files, test_files, sem, classname, train_filenames, test_filenames)
        # read_thread = threading.Thread(target=read_meshes, args=args)
        # read_thread.start()
        out_path = train_img + '/' + classname
        if not os.path.exists(out_path):
            os.mkdir(out_path)

        for filename in train_filenames:
            file = open(filename, 'r')
            lines = file.readlines()
            # sem.acquire()
            # lines = train_files.get()
            # print('filename:', filename, 'out_path', out_path)
            draw(window, filename, out_path, lines)

        out_path = test_img + '/' + classname
        for filename in test_filenames:
            # sem.acquire()
            file = open(filename, 'r')
            lines = file.readlines()
            # lines = test_files.get()
            # print('filename:', filename, 'out_path', out_path)
            draw(window, filename, out_path, lines)

        # read_thread.join()

main()
