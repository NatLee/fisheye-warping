import setuptools

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name='FisheyeWarping',
    author='Nat Lee',
    author_email='natlee.work@gmail.com',
    description='Dewarp top-down fisheye image to panorama and rewarp panorama to fisheye image.',
    keywords='dewarp, rewarp, panorama, fisheye, mesh, cli',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/NatLee/fisheye-warping',
    project_urls={
        'Documentation': 'https://github.com/NatLee/fisheye-warping',
        'Bug Reports': 'https://github.com/NatLee/fisheye-warping/issues',
        'Source Code': 'https://github.com/NatLee/fisheye-warping'
    },
    package_dir={'': 'src'},
    packages=setuptools.find_packages(where='src'),
    package_data={'fisheyewarping': ['py.typed']},
    classifiers=[
        # see https://pypi.org/classifiers/
        'Development Status :: 5 - Production/Stable',

        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Multimedia :: Graphics :: Graphics Conversion',
        'Topic :: Scientific/Engineering :: Image Processing',

        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Programming Language :: Python :: 3.14',
        'Programming Language :: Python :: 3 :: Only',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Typing :: Typed',
    ],
    python_requires='>=3.6',
    install_requires=[
        # OpenCV 4.7+ has no wheels for Python 3.6, so pip would try to compile it from source and fail
        'opencv-python<4.7; python_version < "3.7"',
        'opencv-python; python_version >= "3.7"',
        'numpy',
        'tqdm',
    ],
    extras_require={
        'dev': ['check-manifest'],
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'fisheyewarping=fisheyewarping.cli:main'
        ]
    }
)
