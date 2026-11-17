"""PyPI shows the README without the repository around it, so setup.py turns
every relative link in it into an absolute GitHub link, pinned to the release
tag. The README itself only uses relative links to this repository, so a
release's PyPI page keeps working when files move on the main branch later."""
import pathlib
import re
import runpy
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
SETUP = runpy.run_path(str(ROOT / 'setup.py'), run_name='setup_under_test')
absolute_links = SETUP['absolute_links']

RAW = 'https://raw.githubusercontent.com/NatLee/fisheye-warping/v9.9.9/'
PAGE = 'https://github.com/NatLee/fisheye-warping/blob/v9.9.9/'


class TestReadmeLinks(unittest.TestCase):

    def test_embedded_images_point_at_the_file_in_the_release_tag(self):
        readme = ('<img src="docs/images/round-trip.jpg" width="100%">\n'
                  '![how](docs/images/how-it-works.jpg) <source srcset="docs/images/a.png">\n')
        self.assertEqual(absolute_links(readme, 'v9.9.9'), (
            '<img src="' + RAW + 'docs/images/round-trip.jpg" width="100%">\n'
            '![how](' + RAW + 'docs/images/how-it-works.jpg) <source srcset="' + RAW + 'docs/images/a.png">\n'))

    def test_links_point_at_the_github_page_in_the_release_tag(self):
        readme = ('[`docs/images/101-fisheye.jpg`](docs/images/101-fisheye.jpg) '
                  '[a test](tests/test_readme_formulas.py) <a href="src/fisheyewarping/cli.py">cli</a>')
        self.assertEqual(absolute_links(readme, 'v9.9.9'), (
            '[`docs/images/101-fisheye.jpg`](' + PAGE + 'docs/images/101-fisheye.jpg) '
            '[a test](' + PAGE + 'tests/test_readme_formulas.py) <a href="' + PAGE + 'src/fisheyewarping/cli.py">cli</a>'))

    def test_a_link_around_an_image_is_rewritten_too(self):
        readme = '[![License: MIT](https://img.shields.io/pypi/l/FisheyeWarping.svg)](LICENSE)'
        self.assertEqual(absolute_links(readme, 'v9.9.9'),
                         '[![License: MIT](https://img.shields.io/pypi/l/FisheyeWarping.svg)](' + PAGE + 'LICENSE)')

    def test_absolute_links_and_anchors_are_left_alone(self):
        readme = ('[issue #1](https://github.com/NatLee/fisheye-warping/issues/1) [up](#installation) '
                  '<img src="https://avatars.githubusercontent.com/u/1"> [mail](mailto:a@b.c) '
                  '<a href="/NatLee">x</a> `fisheyewarping --image ./docs/images/101-fisheye.jpg`')
        self.assertEqual(absolute_links(readme, 'v9.9.9'), readme)

    def test_the_long_description_links_to_the_tag_of_this_version(self):
        version = SETUP['VERSION']
        description = SETUP['LONG_DESCRIPTION']
        self.assertIn('raw.githubusercontent.com/NatLee/fisheye-warping/v{}/docs/images/round-trip.jpg'.format(version),
                      description)
        self.assertEqual(absolute_links(description, 'v' + version), description)

    def test_the_readme_links_to_its_own_files_relatively(self):
        # Absolute links to this repository's files would bypass the rewrite above,
        # and would keep pointing at whatever branch they name.
        readme = (ROOT / 'README.md').read_text(encoding='utf-8')
        self.assertEqual(
            re.findall(r'github(?:usercontent)?\.com/NatLee/fisheye-warping/(?:blob|tree|raw|main)/', readme), [])


if __name__ == '__main__':
    unittest.main()
