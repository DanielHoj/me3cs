from setuptools import setup

setup(
    name='me3cs',
    version='0.1.3',
    install_requires=["pandas", "numpy", "scipy"],
    packages=['me3cs', 'me3cs.misc', 'me3cs.metrics', 'me3cs.metrics.regression', 'me3cs.framework',
              'me3cs.framework.helper_classes', 'me3cs.model_types', 'me3cs.model_types.regression',
              'me3cs.model_types.decomposition', 'me3cs.missing_data', 'me3cs.preprocessing', 'me3cs.cross_validation'],
    url='',
    license='',
    author='Daniel Andreas Njust Høj',
    author_email='d.njust.hoej@gmail.com',
    description='Chemometrical package for '
)
