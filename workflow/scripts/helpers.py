import ast
from pathlib import Path

from snakemake.utils import update_config


def literal_eval(s):
    try:
        return ast.literal_eval(s.capitalize())
    except (ValueError, SyntaxError):
        return s


def parse_to_dict(s):
    d = {}
    for item in s.split("-"):
        key, value = item.split("+", 1)
        d[key] = parse_to_dict(value) if "+" in value else literal_eval(value)
    return d


def set_scenario_config(config, wildcards):
    for wc_key, wc_string in wildcards.items():
        if wc_string is not None:
            wc_dict = parse_to_dict(wc_string)
            if wc_key == "st":
                wc_dict = dict(myopic=wc_dict)
            update_config(config, wc_dict)


def mock_snakemake(rulename, configfiles=[], **wildcards):
    """
    This function is expected to be executed from the 'scripts'-directory of '
    the snakemake project. It returns a snakemake.script.Snakemake object,
    based on the Snakefile.

    If a rule has wildcards, you have to specify them in **wildcards.

    Parameters
    ----------
    rulename: str
        name of the rule for which the snakemake object should be generated
    configfiles: list, str
        list of configfiles to be used to update the config
    **wildcards:
        keyword arguments fixing the wildcards. Only necessary if wildcards are
        needed.
    """
    import os

    import snakemake as sm
    from packaging.version import Version, parse
    from pypsa.descriptors import Dict
    from snakemake.script import Snakemake

    script_dir = Path(__file__).parent.resolve()
    root_dir = script_dir.parent

    user_in_script_dir = Path.cwd().resolve() == script_dir
    if user_in_script_dir:
        if root_dir.name == "workflow":
            os.chdir(root_dir.parent)
        else:
            os.chdir(root_dir)
    elif Path.cwd().resolve() != root_dir:
        raise RuntimeError(
            "mock_snakemake has to be run from the repository root"
            f" {root_dir} or scripts directory {script_dir}"
        )
    try:
        for p in sm.SNAKEFILE_CHOICES:
            if os.path.exists(p):
                snakefile = p
                break
        kwargs = (
            dict(rerun_triggers=[]) if parse(sm.__version__) > Version("7.7.0") else {}
        )
        if isinstance(configfiles, str):
            configfiles = [configfiles]

        workflow = sm.Workflow(snakefile, overwrite_configfiles=configfiles, **kwargs)
        workflow.include(snakefile)

        if configfiles:
            for f in configfiles:
                if not os.path.exists(f):
                    raise FileNotFoundError(f"Config file {f} does not exist.")
                workflow.configfile(f)

        workflow.global_resources = {}
        rule = workflow.get_rule(rulename)
        dag = sm.dag.DAG(workflow, rules=[rule])
        wc = Dict(wildcards)
        job = sm.jobs.Job(rule, dag, wc)

        def make_accessable(*ios):
            for io in ios:
                for i in range(len(io)):
                    io[i] = os.path.abspath(io[i])

        make_accessable(job.input, job.output, job.log)
        snakemake = Snakemake(
            job.input,
            job.output,
            job.params,
            job.wildcards,
            job.threads,
            job.resources,
            job.log,
            job.dag.workflow.config,
            job.rule.name,
            None,
        )
        # create log and output dir if not existent
        for path in list(snakemake.log) + list(snakemake.output):
            Path(path).parent.mkdir(parents=True, exist_ok=True)

    finally:
        if user_in_script_dir:
            os.chdir(script_dir)
    return snakemake
