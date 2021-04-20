from qo.qo_utils import ModuleNotFoundIgnore

module_not_found_ignore = ModuleNotFoundIgnore()

source_module_names = [
    'tw'
]

# TODO: automate the following from source_module_names specs
with module_not_found_ignore:
    from qo.tw import *
    from qo import tw

