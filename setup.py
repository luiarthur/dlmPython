from setuptools import setup

setup(name='dlmPython',
      version='0.1',
      description='Dynamic Linear Models',
      url='http://github.com/luiarthur/dlmPython',
      author='Arthur Lui',
      author_email='luiarthur@ucsc.edu',
      license='MIT',
      packages=['dlmPython'],
      install_requires=[ 'numpy==1.8.0', 'scipy' ],
      test_suite='nose.collector',
      tests_require=[ 'nose' ],
      zip_safe=False)
