"""Sphinx configuration for the catsim documentation."""

import datetime
import pathlib
import sys
from importlib.metadata import version as get_version

# Sphinx runs this file from the sphinx/ directory.
sys.path.insert(0, str(pathlib.Path("../src").resolve()))
sys.path.insert(0, str(pathlib.Path("_ext").resolve()))

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
  "sphinx_reredirects",
  "class_tree",
]

myst_enable_extensions = [
  "colon_fence",
  "deflist",
  "dollarmath",
  "fieldlist",
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
bibtex_default_style = "alpha"
bibtex_reference_style = "author_year"
# Each technique page lists its own sources, so one entry appears on several pages and on the
# global Bibliography page. The alpha style gives an entry the same label on every page.
suppress_warnings = ["bibtex.duplicate_citation"]

# Each inventory comes from its project's site, or from the copy stored in _inventories/ when the
# site cannot be reached, so that the strict build does not fail on a network timeout.
# CONTRIBUTING.md ("Building the documentation") gives the command that refreshes the copies.
# A site that stalls rather than refusing the connection falls back to the copy after this many seconds.
intersphinx_timeout = 10
intersphinx_mapping = {
  name: (url, (None, f"_inventories/{name}.inv"))
  for name, url in {
    "python": "https://docs.python.org/3/",
    "numpy": "https://numpy.org/doc/stable/",
    "matplotlib": "https://matplotlib.org/stable/",
    "scipy": "https://docs.scipy.org/doc/scipy/",
  }.items()
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
exclude_patterns = ["_build", "AGENTS.md", "CLAUDE.md", "**/_template.md"]

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
  # The navbar shows the seven top-level entries of the table of contents without a "More" menu.
  "header_links_before_dropdown": 7,
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

# A page that moved, merged into another or left the site keeps its old address: the build writes
# a small page there that sends the reader to the topic's current home. Each target is relative to
# the old page's folder, because the site is served below /catsim/. Old ADR addresses are matched
# by their full file name: an old address and a current ADR can share a number.
redirects = {
  # 0.21's module pages, at the site's root.
  "cat": "api/metrics.html",
  "estimation": "api/techniques/estimation/index.html",
  "initialization": "api/techniques/initialization/index.html",
  "irt": "api/irt.html",
  "plot": "api/plot.html",
  "selection": "api/techniques/selection/index.html",
  "simulation": "api/simulation.html",
  "stopping": "api/techniques/stopping/index.html",
  "api/cat": "metrics.html",
  # Technique pages merged into another page or renamed.
  "api/techniques/initialization/fixed-point-initializer": "index.html",
  "api/techniques/initialization/random-initializer": "index.html",
  "api/techniques/selection/a-strat-b-block-selector": "stratified-selector.html",
  "api/techniques/selection/a-strat-selector": "stratified-selector.html",
  "api/techniques/selection/max-info-b-block-selector": "stratified-selector.html",
  "api/techniques/selection/max-info-strat-selector": "stratified-selector.html",
  "api/techniques/stopping/min-error-stopper": "standard-error-stopper.html",
  "api/techniques/stopping/test-length-stopper": "max-length-stopper.html",
  # Planned techniques, whose specifications are in specs/ on GitHub.
  "api/techniques/estimation/owen-estimator": "../../../planned-techniques.html",
  "api/techniques/selection/klp-selector": "../../../planned-techniques.html",
  "api/techniques/selection/mepv-selector": "../../../planned-techniques.html",
  "api/techniques/selection/mpwi-selector": "../../../planned-techniques.html",
  "api/techniques/selection/sympson-hetter-exposure": "../../../planned-techniques.html",
  "api/techniques/stopping/glr-stopper": "../../../planned-techniques.html",
  "api/techniques/stopping/kl-stopper": "../../../planned-techniques.html",
  "api/techniques/stopping/min-info-bank-stopper": "../../../planned-techniques.html",
  "api/techniques/stopping/predicted-sem-stopper": "../../../planned-techniques.html",
  "api/techniques/stopping/sprt-stopper": "../../../planned-techniques.html",
  "api/techniques/stopping/stochastic-curtailment-stopper": "../../../planned-techniques.html",
  # ADRs of earlier numberings, mapped to their successors in the current set, and the two whose
  # content is in the contributing guide.
  "adr/0001-explicit-cat-engine": "0006-steps-from-explicit-inputs.html",
  "adr/0001-unidimensional-dichotomous-scope": "0002-unidimensional-dichotomous-scope.html",
  "adr/0002-four-parameter-item-matrix": "0003-four-parameter-item-matrix.html",
  "adr/0002-remove-legacy-simulator-apis": "0006-steps-from-explicit-inputs.html",
  "adr/0003-explicit-cat-engine": "0006-steps-from-explicit-inputs.html",
  "adr/0003-runtime-state-ownership": "0012-exposure-rates.html",
  "adr/0004-explicit-strategy-contracts": "0007-strategy-contracts.html",
  "adr/0004-runtime-state-ownership": "0012-exposure-rates.html",
  "adr/0005-explicit-strategy-contracts": "0007-strategy-contracts.html",
  "adr/0005-immutable-step-results": "0006-steps-from-explicit-inputs.html",
  "adr/0006-public-import-paths-and-names": "0023-public-import-paths-and-names.html",
  "adr/0006-versioned-tutorial-notebooks": "../contributing.html",
  "adr/0007-how-a-session-ends": "0010-how-a-session-ends.html",
  "adr/0007-release-gated-by-docs": "../contributing.html",
  "adr/0008-random-tie-breaking": "0017-random-tie-breaking.html",
  "adr/0008-session-exclusions-and-exposure-control": "0011-exclusions-and-exposure-control.html",
  "adr/0009-ability-scale-and-extreme-patterns": "0018-ability-scale-and-extreme-patterns.html",
  "adr/0009-immutable-step-results": "0006-steps-from-explicit-inputs.html",
  "adr/0010-published-progressive-schedule": "0019-published-progressive-schedule.html",
  "adr/0010-random-numbers-and-reproducibility": "0013-random-numbers-and-reproducibility.html",
  "adr/0011-how-a-session-ends": "0010-how-a-session-ends.html",
  "adr/0011-random-tie-breaking": "0017-random-tie-breaking.html",
  "adr/0012-ability-scale-and-extreme-patterns": "0018-ability-scale-and-extreme-patterns.html",
  "adr/0012-unidimensional-dichotomous-scope": "0002-unidimensional-dichotomous-scope.html",
  "adr/0013-published-progressive-schedule": "0019-published-progressive-schedule.html",
  "adr/0014-plotting-as-an-optional-extra": "0022-plotting-as-an-optional-extra.html",
  "adr/0014-public-import-paths-and-names": "0023-public-import-paths-and-names.html",
  "adr/0015-plotting-as-an-optional-extra": "0022-plotting-as-an-optional-extra.html",
  "adr/0016-session-exclusions-and-exposure-control": "0011-exclusions-and-exposure-control.html",
}

numpydoc_show_class_members = False

# Every cross-reference must resolve (sphinx-build -n). NumPy's array aliases evaluate to classes
# under private module paths that NumPy's inventory does not list, and a module that imports
# numpy.typing for the type checker only leaves its annotations as the text "npt.ArrayLike".
nitpicky = True
nitpick_ignore = [
  ("py:class", "numpy._typing._array_like.NDArray"),
  ("py:class", "numpy._typing._array_like.ArrayLike"),
  ("py:class", "npt.ArrayLike"),
]

# The build fails for a public callable whose docstring leaves out a parameter (numpydoc's PR01).
# Ruff's DOC rules check the sections a docstring has against the code, not that they exist.
numpydoc_validation_checks = {"PR01"}
# A class documents its constructor's parameters; autodoc renders the class docstring alone, so
# __init__ is never checked.
numpydoc_validation_exclude = {
  # numpydoc reads the call signature of an Enum's metaclass as the class's
  # parameters, and that of an array alias as the alias's.
  r"^catsim\.(IrtModel|StopReason)$",
  r"^catsim\.selection\.(CapFallback|ClusterMethod)$",
  r"^catsim\.plot\.PlotType$",
  r"^catsim\.typing\.(FloatArray|IntArray)$",
  # Autodoc reads the docstrings of the names a module imports before it leaves them off the page.
  r"^catsim\.irt\.overload$",
  r"^catsim\.plot\.(Axes|MaxNLocator|StrEnum|assert_never)$",
  r"^catsim\.typing\.Any$",
}
