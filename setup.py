from setuptools import setup, find_packages


setup(
    name='me3cs',
    version='0.1.3',
    license='BSD 3-clause',
    author='Daniel Andreas Njust Høj',
    author_email='d.njust.hoej@gmail.com',
    packages=find_packages('me3cs'),
    package_dir={'': 'me3cs'},
    url='https://github.com/DanielHoj/me3cs',
    keywords='chemometrics',
    install_requires=["pandas",
                      "numpy",
                      "scipy"],

)

