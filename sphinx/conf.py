"""Sphinx configuration for the catsim documentation; it also regenerates README.md from its sources."""

import datetime
import pathlib
import sys
from importlib.metadata import version as get_version

# Sphinx runs this file from the sphinx/ directory.
sys.path.insert(0, str(pathlib.Path("../src").resolve()))
sys.path.insert(0, str(pathlib.Path("_ext").resolve()))

filenames = ["readme_head.md", "readme_body.md"]
with pathlib.Path("../README.md").open("w", encoding="utf-8") as outfile:
  outfile.writelines(pathlib.Path(fname).read_text(encoding="utf-8") for fname in filenames)

# -- General configuration ------------------------------------------------

extensions = [
  "matplotlib.sphinxext.plot_directive",
  "sphinx.ext.autodoc",
  "sphinx.ext.githubpages",
  "sphinx.ext.intersphinx",
  "sphinx.ext.mathjax",
  "sphinx.ext.viewcode",
  "numpydoc",
  "myst_parser",
  "sphinxcontrib.bibtex",
  "sphinxcontrib.mermaid",
  "class_tree",
]

myst_enable_extensions = [
  "colon_fence",
  "deflist",
  "dollarmath",
]

# Mermaid diagrams are drawn in the reader's browser, and sphinxcontrib-mermaid redraws them
# when the page switches between light and dark mode. Mermaid writes each diagram's natural
# size into the SVG (useMaxWidth off), which _static/mermaid.css keeps unless the diagram is
# wider than the page. Class trees leave out UML's empty member boxes, since they only show
# inheritance.
mermaid_light_theme = "neutral"
mermaid_dark_theme = "dark"
mermaid_init_config = {
  "startOnLoad": False,
  "flowchart": {"useMaxWidth": False},
  "class": {"useMaxWidth": False, "hideEmptyMembersBox": True},
  # Mermaid 11's class diagram renderer reads useMaxWidth from the state diagram settings.
  "state": {"useMaxWidth": False},
}

# The plot directive embeds the first format in the page, so plots are vector SVG, sharp on any
# screen. The tight bounding box keeps labels that sit outside the axes, such as a twin y axis
# label, inside the image. The links to other formats and to the plot's source are left out.
plot_formats = [("svg", 100)]
plot_rcparams = {"savefig.bbox": "tight"}
plot_html_show_formats = False
plot_html_show_source_link = False

bibtex_bibfiles = ["references.bib"]
bibtex_default_style = "plain"
bibtex_reference_style = "author_year"

intersphinx_mapping = {
  "python": ("https://docs.python.org/3", None),
  "numpy": ("https://numpy.org/doc/stable/", None),
  "scipy": ("https://docs.scipy.org/doc/scipy/", None),
  "matplotlib": ("https://matplotlib.org/stable/", None),
}

source_suffix = {
  ".md": "markdown",
  ".rst": "restructuredtext",
}

master_doc = "index"

project = "catsim"
copyright = f"2015-{datetime.datetime.now(tz=datetime.UTC).year}, Douglas De Rizzo Meneghetti"  # noqa: A001
author = "Douglas De Rizzo Meneghetti"

# |version| and |release| come from the installed package's metadata.
version = get_version("catsim")
release = version

language = "en"

# AGENTS.md and CLAUDE.md are instructions for agents and contributors, and the templates are copied,
# not read: none of them is a page.
exclude_patterns = ["_build", "readme_head.md", "readme_body.md", "AGENTS.md", "CLAUDE.md", "**/_template.md"]

pygments_style = "sphinx"

numfig = True
numfig_secnum_depth = 0

# -- Options for HTML output ----------------------------------------------

html_theme = "pydata_sphinx_theme"

html_theme_options = {
  "github_url": "https://github.com/douglasrizzo/catsim",
  "show_toc_level": 2,
  "navigation_depth": 2,
  "collapse_navigation": True,
  "show_nav_level": 2,
  "navbar_end": ["navbar-icon-links", "theme-switcher"],
  "logo": {
    "image_light": "logo/horizontal-light.svg",
    "image_dark": "logo/horizontal-dark.svg",
  },
}

# The navbar logo is set per color mode in html_theme_options["logo"].

# The favicon is one file shown on the browser's own tab bar, whatever the page's
# color mode, so it uses the tile version of the mark, which carries its own background.
html_favicon = "logo/favicon.svg"

html_static_path = ["_static"]
html_css_files = ["mermaid.css"]

numpydoc_show_class_members = False
