from setuptools import setup, Extension
from Cython.Build import cythonize

# IAIN had to change all references to be relative to sigdirect
setup(
	ext_modules=cythonize(Extension(
		name="sigdirect",
		sources=["sigdirect/src/sigdirect.cpp", "sigdirect/src/rule.cpp", "sigdirect/src/node.cpp",
				 "sigdirect/src/rule_node.cpp", "sigdirect/mysigdirect.pyx", ],
		language="c++",
		include_dirs=["sigdirect", "sigdirect/include", "sigdirect/src", "sigdirect/lib", "sigdirect/tests", "sigdirect/lib/plog/include"],
		extra_compile_args=['-O3',  '-std=c++17'],
)))
