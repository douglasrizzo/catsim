from setuptools import setup, find_packages

setup(name='catsim',
      description='Computerized Adaptive Testing' +
                  ' assisted by Similarity Algorithm',
      author='Douglas De Rizzo Meneghetti',
      author_email='douglasrizzom@gmail.com',
      url='https://www.github.com/douglasrizzo/catsim',
      packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
      install_requires=[
          'numpy', 'scipy', 'scikit-learn', 'pandas'],
      license='GPLv2'
      )
