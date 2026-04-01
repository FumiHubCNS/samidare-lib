from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Dict


def configure_spark_env_from_pixi(
    *,
    set_process_env: bool = True,
    prefer_pyspark_bundle: bool = True,
) -> Dict[str, str]:
    """
    Detect Spark/Java locations inside the current Pixi Python environment
    and return environment variables suitable for PySpark.

    Assumptions:
    - This function is executed by Python inside a Pixi environment.
    - PySpark may be installed from PyPI inside the current environment.
    - Java is expected to be available in the Pixi environment's PATH.

    Behavior:
    - Finds JAVA_HOME from:
        1. current process JAVA_HOME if valid
        2. <env>/lib/jvm
        3. inferred from `which java`
    - Finds SPARK_HOME from:
        1. bundled PySpark directory if it contains bin/spark-submit
        2. existing SPARK_HOME only if valid
        3. otherwise unsets SPARK_HOME
    """

    env_updates: Dict[str, str] = {}

    python_path = Path(sys.executable).resolve()
    env_prefix = python_path.parent.parent  # .../.pixi/envs/default

    # ----------------------------
    # Resolve JAVA_HOME
    # ----------------------------
    candidate_java_homes: list[Path] = []

    current_java_home = os.environ.get("JAVA_HOME")
    if current_java_home:
        candidate_java_homes.append(Path(current_java_home))

    candidate_java_homes.append(env_prefix / "lib" / "jvm")

    java_bin = shutil.which("java")
    if java_bin:
        java_bin_path = Path(java_bin).resolve()
        # Usually: <env>/bin/java -> JAVA_HOME often lives in <env>/lib/jvm
        candidate_java_homes.append(java_bin_path.parent.parent / "lib" / "jvm")
        candidate_java_homes.append(java_bin_path.parent.parent)

    java_home: Path | None = None
    for candidate in candidate_java_homes:
        if candidate.exists():
            java_home = candidate
            break

    if java_home is not None:
        env_updates["JAVA_HOME"] = str(java_home)

    # ----------------------------
    # Resolve SPARK_HOME
    # ----------------------------
    spark_home: Path | None = None

    if prefer_pyspark_bundle:
        try:
            import pyspark  # type: ignore

            pyspark_dir = Path(pyspark.__file__).resolve().parent
            spark_submit = pyspark_dir / "bin" / "spark-submit"
            if spark_submit.exists():
                spark_home = pyspark_dir
        except Exception:
            spark_home = None

    if spark_home is None:
        current_spark_home = os.environ.get("SPARK_HOME")
        if current_spark_home:
            candidate = Path(current_spark_home)
            if (candidate / "bin" / "spark-submit").exists():
                spark_home = candidate

    if spark_home is not None:
        env_updates["SPARK_HOME"] = str(spark_home)
    else:
        # Important: remove it instead of setting empty string
        if set_process_env:
            os.environ.pop("SPARK_HOME", None)

    # ----------------------------
    # Apply to current process
    # ----------------------------
    if set_process_env:
        for key, value in env_updates.items():
            os.environ[key] = value

    return env_updates

def main():
    envs = configure_spark_env_from_pixi()
    print(envs)
    print("JAVA_HOME =", os.environ.get("JAVA_HOME"))
    print("SPARK_HOME =", os.environ.get("SPARK_HOME"))

    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder
        .master("local[*]")
        .appName("samidare-test")
        .getOrCreate()
    )
    print(spark.version)

if __name__ == "__main__":
    main()