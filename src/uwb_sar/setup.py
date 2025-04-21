from setuptools import find_packages, setup
import os 
from glob import glob
package_name = 'uwb_sar'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Charith Premachandra',
    maintainer_email='gihan_appuhamilage@mymail.sutd.edu.sg',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'plot_node = uwb_sar.plot_raw_obs:main',
            'sar_node = uwb_sar.gen_sar:main',
            'fm_node = uwb_sar.feature_match:main',
        ],
    },
)
