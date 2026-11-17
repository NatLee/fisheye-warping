import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
#: owner/name on GitHub. The project URLs and the README links on PyPI point here.
REPOSITORY = 'NatLee/fisheye-warping'
REPO_URL = 'https://github.com/' + REPOSITORY
RAW_FILES = 'https://raw.githubusercontent.com/' + REPOSITORY

# setup.cfg reads the version from the package for the release itself; it is read here
# too, only to know which tag the README links on PyPI should point at.
with open(os.path.join(HERE, 'src', 'fisheyewarping', '__init__.py'), 'r', encoding='utf-8') as fh:
    VERSION = re.search(r'''^__version__ = ['"]([^'"]+)['"]''', fh.read(), re.MULTILINE).group(1)
#: Releases are tagged v<version> (see `git tag`), so this is the tree the release was built from.
RELEASE_TAG = 'v' + VERSION

# A link target that is relative to the repository: no scheme ("https:", "mailto:"),
# no in-page anchor and no site-absolute path.
_RELATIVE = r'(?![A-Za-z][A-Za-z0-9+.-]*:|[#/])'
# Embedded files (HTML src/srcset, Markdown images) need the raw file. They are rewritten
# first, so every relative target that is left after "](" or href=" is a link ...
_EMBEDDED = re.compile(r'(\b(?:src|srcset)="|!\[[^\]]*\]\()' + _RELATIVE)
# ... to the GitHub page of the file or folder, including a link around an image: [![x](…)](LICENSE).
_LINKED = re.compile(r'(\bhref="|\]\()' + _RELATIVE)


def absolute_links(markdown, ref):
    """Point every relative link in ``markdown`` at this repository's files in ``ref``.

    The README uses relative links, which work on GitHub and in a clone. PyPI shows
    the text on its own, where they would all be broken. Pinning them to the release
    tag keeps a release's page intact when files later move on the main branch.
    """
    markdown = _EMBEDDED.sub(r'\g<1>{}/{}/'.format(RAW_FILES, ref), markdown)
    return _LINKED.sub(r'\g<1>{}/blob/{}/'.format(REPO_URL, ref), markdown)


with open(os.path.join(HERE, 'README.md'), 'r', encoding='utf-8') as fh:
    LONG_DESCRIPTION = absolute_links(fh.read(), RELEASE_TAG)

if __name__ == '__main__':
    # setuptools runs this file as __main__. The guard lets the tests import the
    # functions above without installing setuptools.
    import setuptools

    setuptools.setup(
        name='FisheyeWarping',
        author='Nat Lee',
        author_email='natlee.work@gmail.com',
        description='Dewarp top-down fisheye image to panorama and rewarp panorama to fisheye image.',
        keywords='dewarp, rewarp, panorama, fisheye, mesh, cli',
        long_description=LONG_DESCRIPTION,
        long_description_content_type='text/markdown',
        url=REPO_URL,
        project_urls={
            'Documentation': REPO_URL,
            'Bug Reports': REPO_URL + '/issues',
            'Source Code': REPO_URL,
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
        python_requires='>=3.8',  # Python 3.6 and 3.7 users get 1.0.2, the last release for them
        install_requires=[
            'opencv-python-headless',  # the same cv2 without the window functions, which this tool never uses
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
