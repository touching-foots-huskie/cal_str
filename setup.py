from setuptools import setup, find_packages
setup(
    name = 'cal_str',
    version = '0.1',
    packages = find_packages('src'),
    package_dir = {'':'src'},
    package_data = {
        'cal_str' :
        [
        'envs/assets/*.xml',
        ]
    },
)
