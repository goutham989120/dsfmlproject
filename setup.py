from setuptools import setup, find_packages
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path:str) -> List[str]:
    ''' this function will return the list of requirements '''
    requirements = []
    with open(file_path, 'r') as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements


setup(
    name='dsfmlproject',
    version='1.0.0',
    author='Goutham S M',
    author_email='goutham989120@gmail.com',
    packages=find_packages(),
    # install_requires=['pandas', 'numpy', 'scikit-learn', 'matplotlib', 'seaborn']
    install_requires=get_requirements('requirements.txt')
)