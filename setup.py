from setuptools import setup, Extension
import platform

c_args = []
if platform.system() == "Windows":
    c_args = ['/O2', '/arch:AVX2']
else:
    c_args = ['-O3', '-march=native', '-funroll-loops']

module = Extension('c_sequence',
                    sources=['src/c_game/game.c'],
                    extra_compile_args=c_args)

setup(name='c_sequence',
      version='1.0',
      description='C implementation of Sequence game logic',
      ext_modules=[module])
