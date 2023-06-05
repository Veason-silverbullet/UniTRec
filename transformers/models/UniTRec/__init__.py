from ...utils import OptionalDependencyNotAvailable, is_torch_available


_import_structure = {"configuration_unitrec": ["UniTRecConfig"]}

if not is_torch_available():
    raise OptionalDependencyNotAvailable()
else:
    _import_structure["modeling_unitrec"] = ["UniTRecModel", "UniTRecPretrainedModel"]

from .configuration_unitrec import UniTRecConfig

if not is_torch_available():
    raise OptionalDependencyNotAvailable()
else:
    from .modeling_unitrec import UniTRecModel, UniTRecPretrainedModel
