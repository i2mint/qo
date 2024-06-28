"""
A medley of my most frequently used tools.
"""

from contextlib import suppress
from functools import partial

from qo.qo_utils import module_not_found_ignore

from warnings import warn

from collections import Counter, defaultdict
from datetime import datetime
from pprint import PrettyPrinter
import json

try:
    import matplotlib.pylab as plt
except ModuleNotFoundError as e:
    warn(f'{e}: {e.args}')

ddir = lambda o: [a for a in dir(o) if not a.startswith('_')]
dddir = lambda o: [a for a in dir(o) if not a.startswith('__')]


def notebook_url(path: str, root_url='http://localhost:8888/'):
    import os

    # TODO: Figure out how to get the rootdir from the root_url
    if path.endswith('.ipynb'):
        return os.path.join(root_url, path)
    else:
        return os.path.join(root_url, 'tree', path)


def goto_definition_file(obj):
    """Will open the file that defines the obj and prints"""
    import inspect, subprocess

    _, lineno = inspect.findsource(obj)
    subprocess.run(['open', inspect.getfile(obj)])
    print(f'Definition on line number: {lineno}')


def goto_definition(
    obj, command_template='open -na "PyCharm.app" --args --line {lineno} "{filepath}"'
):
    """Opens the definition of the object in the file it was defined at the line it
    was defined.

    The default is set to use pycharm to open the file on a mac.

    To customize for your system/file_viewer, just use `functools.partial` to
    fix the `command_template` for your case.

    For pycharm on other systems, see:
    https://www.jetbrains.com/help/pycharm/opening-files-from-command-line.html

    For vi, see:
    https://www.cyberciti.biz/faq/linux-unix-command-open-file-linenumber-function/

    Etc.
    """
    import inspect, os

    try:
        filepath = inspect.getfile(obj)
        _, lineno = inspect.findsource(obj)
        command = command_template.format(filepath=filepath, lineno=lineno)
        os.system(command)  # would prefer to use subprocess, but something I'm missing
        ## Not working with subprocess.run
        #     import subprocess
        #     print(command)
        #     return subprocess.run(command.split(' '))
    except TypeError:
        if hasattr(obj, 'func'):
            return goto_definition(obj.func, command_template)
        else:
            raise


ignore_errors = suppress(ModuleNotFoundError, ImportError, RuntimeError)

with module_not_found_ignore:
    from scraped import download_site, markdown_of_site

with module_not_found_ignore:
    from hubcap import github_repo_text_aggregate

with module_not_found_ignore:
    from tested import validate_codec

with module_not_found_ignore:
    from unbox import print_missing_names

with module_not_found_ignore:
    from meshed import code_to_dag, DAG, FuncNode

with module_not_found_ignore:
    from tabled import get_tables_from_url

with module_not_found_ignore:

    def disp_wfsr(
        wf,
        sr=44100,
        offset_s=None,  # for display of x axix ticks only (not implemented yet)
    ):
        try:
            import matplotlib.pylab as plt
            from IPython.display import Audio

            plt.plot(
                wf
            )  # TODO: mk x axis ticks be labeled based on time (aligned to whole seconds, or minutes...)
            return Audio(data=wf, rate=sr)
        except (ImportError, ModuleNotFoundError):
            pass


with module_not_found_ignore:
    from py2store import (
        ihead,
        kvhead,
        QuickStore,
        LocalBinaryStore,
        LocalJsonStore,
        LocalPickleStore,
        LocalTextStore,
    )
    from py2store import kv_wrap, wrap_kvs, filt_iter, cached_keys
    from py2store.util import lazyprop
    from py2store.my.grabbers import grabber_for as _grabber_for

    igrab = _grabber_for('ipython')

with module_not_found_ignore:
    from i2.doc_mint import doctest_string_print, doctest_string

with module_not_found_ignore:
    import ut.daf.ch
    import ut.daf.manip
    import ut.daf.gr
    import ut.daf.to
    from ut.daf.diagnosis import diag_df as diag_df

with module_not_found_ignore:
    import ut.pdict.get

with module_not_found_ignore:
    import ut.util.pstore


with module_not_found_ignore:
    from ut.pcoll.num import numof_trues

with module_not_found_ignore:
    from ut.util.log import printProgress, print_progress

with module_not_found_ignore:
    import ut.pplot.distrib

with module_not_found_ignore:
    from ut.pplot.matrix import heatmap
    from ut.pplot.my import vlines

with module_not_found_ignore:
    import numpy as np

with module_not_found_ignore:
    import pandas as pd

with module_not_found_ignore:
    from ut.sh.py import add_to_pythonpath_if_not_there

with module_not_found_ignore:
    from ut.util.importing import import_from_dot_string

with module_not_found_ignore:
    from ut.net.viz import dgdisp, dagdisp, dot_to_ascii

with module_not_found_ignore:
    from ut.util.ipython import all_table_of_contents_html_from_notebooks


with ignore_errors:
    from ut.webscrape.tables import get_tables_from_url

with ignore_errors:
    from i2.deco import (
        preprocess,
        postprocess,
        preprocess_arguments,
        input_output_decorator,
    )
    from i2.deco import wrap_class_methods_input_and_output
    from i2.signatures import Sig

with ignore_errors:
    from ut.util.my_proj_populate import populate_proj_from_url


with ignore_errors:
    from ut.util.context_managers import TimerAndFeedback

with ignore_errors:
    import sklearn
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA, IncrementalPCA
    from sklearn.cluster import KMeans
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.neighbors import (
        NearestNeighbors,
        KNeighborsClassifier,
        KNeighborsTransformer,
        KNeighborsRegressor,
    )
    from sklearn.feature_extraction.text import TfidfVectorizer


with ignore_errors:
    from grub import CodeSearcher

    def grub_code(query, module):
        search = CodeSearcher(module).fit()
        return search(query)
