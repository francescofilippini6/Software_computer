from setuptools import setup, find_packages
# from pkg_resources import get_distributio
with open('requirements.txt') as fobj:
    requirements = [l.strip() for l in fobj.readlines()]
setup(
    name='ARCA-CNN',
    description=
                'events in the neutrino telescope KM3NeT',
    
    author='Francesco Filippini',
    author_email='francesco.filippini6@studio.unibo.it',
    license='AGPL',
    install_requires=requirements,
    packages=find_packages(),
    include_package_data=True,
   
    setup_requires=['setuptools_scm'],
   
   
)

__author__ = 'Francesco Filippini'
