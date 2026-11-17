"""Bundle the source and dependencies of the Zia predictor for MLflow."""

import importlib.metadata
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

PYFUNC_LOADER = "zia_mlflow_loader"


def pyfunc_dependencies(model_name):
    packages = ("numpy", "scipy", "mlflow-skinny", "pandas", "river", "pydantic", "PyYAML")
    requirements = [f"{package}=={importlib.metadata.version(package)}" for package in packages]
    if model_name == "xgboost":
        import xgboost

        version = xgboost.__version__
        marker = "sys_platform == 'linux' and platform_machine == 'x86_64'"
        requirements.extend(
            [
                f"xgboost-cpu=={version}; {marker}",
                f"xgboost=={version}; sys_platform != 'linux' or platform_machine != 'x86_64'",
            ]
        )
    return requirements


def pyfunc_code_paths(directory):
    """Package the loader and its source together without one upload per code file."""
    package = Path(__file__).resolve().parents[1]
    archive = Path(directory) / "continual_bench.zip"
    with ZipFile(archive, "w", compression=ZIP_DEFLATED) as bundle:
        for source in sorted(package.rglob("*.py")):
            bundle.write(source, source.relative_to(package.parent))
    return [str(package / "models" / f"{PYFUNC_LOADER}.py"), str(archive)]
