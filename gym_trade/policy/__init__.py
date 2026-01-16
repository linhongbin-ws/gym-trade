import pkgutil
import importlib

def _import_all_submodules(package_name, package_path):
    for _, module_name, is_pkg in pkgutil.iter_modules(package_path):
        full_name = f"{package_name}.{module_name}"
        module = importlib.import_module(full_name)

        if is_pkg and hasattr(module, "__path__"):
            _import_all_submodules(full_name, module.__path__)

_import_all_submodules(__name__, __path__)
