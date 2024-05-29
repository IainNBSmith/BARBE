from setuptools import setup, Extension
from Cython.Build import cythonize

# IAIN had to change all references to be relative to sigdirect
setup(
	ext_modules=cythonize(Extension(
		name="sigdirect",
		sources=["src/sigdirect.cpp", "src/rule.cpp", "src/node.cpp", "src/rule_node.cpp", "mysigdirect.pyx", ],
		language="c++",
		include_dirs=[".", "include", "src", "lib", "tests", "lib/plog/include"],
		extra_compile_args=['-O3',  '-std=c++17'],
)))
