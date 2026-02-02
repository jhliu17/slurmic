# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import importlib
import inspect
import pathlib
import subprocess
from typing import Union

import slurmic

project = "slurmic"
copyright = "2025"
author = "Junhao Liu"
repository = "https://github.com/jhliu17/slurmic"
commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.githubpages",
    "sphinx.ext.linkcode",
    # "sphinx_autodoc_typehints",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "myst_parser",
    "sphinx_design"
]
myst_enable_extensions = ["colon_fence"]

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
}
autodoc_inherit_docstrings = False
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"
autodoc_typehints_format = "short"

autosummary_ignore_module_all = False


def linkcode_resolve(domain: str, info: dict) -> Union[str, None]:
    module = info.get("module", "")
    fullname = info.get("fullname", "")

    if not module or not fullname:
        return None

    try:
        objct = importlib.import_module(module)
        for name in fullname.split("."):
            objct = getattr(objct, name)

        while hasattr(objct, "__wrapped__"):
            objct = objct.__wrapped__

        file = inspect.getsourcefile(objct)
        if file is None:
            return None

        file_path = pathlib.Path(file).relative_to(root)
        lines, start = inspect.getsourcelines(objct)
        end = start + len(lines) - 1

        return f"{repository}/blob/{commit}/{file_path}#L{start}-L{end}"
    except Exception:
        return None


html_theme = "furo"
html_static_path = ["_static"]
templates_path = ["_templates"]
html_css_files = [
    "custom.css",
]
# html_logo = "./_static/apple-touch-icon-white.png"
# html_favicon = "./_static/favicon.ico"
html_title = "slurmic"
html_show_sourcelink = True
html_theme_options = {
    # "light_logo": "apple-touch-icon.png",
    # "dark_logo": "apple-touch-icon.png",
    # "sidebar_hide_name": True,
    "source_repository": "https://github.com/jhliu17/slurmic",
    "source_branch": "main",
    "source_directory": "docs/",
    "top_of_page_buttons": ["view"],
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/jhliu17/slurmic",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,
            "class": "",
        },
    ],
}
