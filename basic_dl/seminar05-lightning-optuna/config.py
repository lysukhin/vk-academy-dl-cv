"""Tools for configuration using default config.

All configurable classes must have :meth:`get_default_config` static method
which returns dictionary of default values. Than you can use
:func:`prepare_config` function to construct actual config. Actual config
can be ``None``, ``dict`` or ``str`` containing path to the file.

**Example**::

    from dalib.config import prepare_config

    class Configurable():
        @staticmethod
        def get_default_config():
            return OrderedDict([
                ("arg1", 10),
                ("arg2", None)
            ])

        def __init__(self, *args, config=None):
            config = prepare_config(self, config)
            self.arg1 = config["arg1"]
            self.arg2 = config["arg2"]

    obj = Configurable(config={"arg1": 5})
    print(obj.arg1)  # 5
    print(obj.arg2)  # None

Config files use YAML syntax. The special key `_type` can be used in configs to specify
target class. If types are provided, they are checked during initialization.

**Example**::

    system:
        subsystem:
            _type: SubsystemClass
            arg1: [5.0, 2.0]
"""

import threading
from collections import OrderedDict
from copy import deepcopy

import optuna
import yaml


CONFIG_TYPE = "_type"
CONFIG_HYPER = "_hyper"


class ConfigError(Exception):
    """Exception class for errors in config."""
    pass


def read_config(filename):
    with open(filename) as fp:
        return yaml.safe_load(fp)


def write_config(config, filename):
    with open(filename, "w") as fp:
        yaml.dump(config, fp)


def _check_hyper(type, choices=None, min=None, max=None, step=None, log=None):
    pass


class suggest_config:
    trials = {}

    def __init__(self, trial):
        thread = threading.get_ident()
        if thread in suggest_config.trials:
            raise RuntimeError("Trial overwrite.")
        suggest_config.trials[thread] = trial

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        thread = threading.get_ident()
        if thread not in suggest_config.trials:
            raise RuntimeError("Trial lost.")
        del suggest_config.trials[thread]

    @staticmethod
    def get_trial():
        thread = threading.get_ident()
        if thread not in suggest_config.trials:
            raise RuntimeError("To sample hyperparameters use `suggest_config` context.")
        return suggest_config.trials[thread]


class Hyper:
    def __init__(self, name, type=None, choices=None, low=None, high=None, step=None, log=None):
        if (choices is not None) and ((low is not None) or (high is not None) or (log is not None) or (step is not None)):
            raise ValueError("Can't mix choices with low, high, step and log parameters.")
        if (type is None) and (choices is None):
            raise ValueError("Type or choices for hyperparameter must be provided.")

        numeric_params = {
            "low": low,
            "high": high,
            "step": step,
            "log": log
        }
        if type in {"str", "bool"}:
            for k, v in numeric_params.items():
                if v is not None:
                    raise ValueError("Can't use {} with type {}.".format(k, type))
            if (type == "str") and (choices is None):
                raise ValueError("Choices must be provided for str hyper parameter.")
            if (type == "bool") and (choices is not None):
                raise ValueError("Can't use choices with type bool.")
        elif type not in {None, "int", "float"}:
            raise ValueError("Unknown hyperparameter type: {}.".format(type))
        self._config = {
            "name": name,
            "type": type,
            "choices": choices
        }
        self._config.update(numeric_params)

    @property
    def config(self):
        return self._config

    def sample(self):
        trial = suggest_config.get_trial()
        if self._config["choices"] is not None:
            return trial.suggest_categorical(self._config["name"], self._config["choices"])
        elif self._config["type"] == "bool":
            return trial.suggest_categorical(self._config["name"], [True, False])
        elif self._config["type"] == "int":
            kwargs = {k: v for k, v in self._config.items()
                      if (k not in {"type"}) and (v is not None)}
            return trial.suggest_int(**kwargs)
        elif self._config["type"] == "float":
            kwargs = {k: v for k, v in self._config.items()
                      if (k not in {"type"}) and (v is not None)}
            return trial.suggest_float(**kwargs)
        else:
            assert False, "Unexpected type: {}.".format(self._config["type"])


def _propagate_hyper_names(config, prefix=None):
    """Set hyperparameters names relative to the config root."""
    if not isinstance(config, dict):
        return
    prefix = "" if prefix is None else prefix + "."
    if CONFIG_HYPER in config:
        hyper = config[CONFIG_HYPER]
        if not isinstance(hyper, dict):
            raise ConfigError("{} must be dict, got {}.".format(CONFIG_HYPER, type(hyper)))
        new_hyper = {}
        for k, v in hyper.items():
            if v is None:
                v = {}
            elif not isinstance(v, dict):
                raise ConfigError("Hyperparameter description must be dict or None, got {}.".format(type(v)))
            new_hyper[k] = v
            if v.get("name", None) is None:
                v["name"] = prefix + k
        config[CONFIG_HYPER] = new_hyper
    for k, v in config.items():
        if k in {CONFIG_TYPE, CONFIG_HYPER}:
            continue
        _propagate_hyper_names(v, prefix + k)


def _is_hyper_only_config(config):
    num_hyper_only_subconfigs = 0
    for k, v in config.items():
        if k in {CONFIG_HYPER, CONFIG_TYPE}:
            continue
        if (not isinstance(v, dict)) or (not _is_hyper_only_config(v)):
            return False
        num_hyper_only_subconfigs += 1
    return (num_hyper_only_subconfigs > 0) or (CONFIG_HYPER in config)


def prepare_config(self, config=None):
    """Set defaults and check fields.

    Config is a dictionary of values. Method creates new config using
    default class config. Result config keys are the same as default config keys.

    Args:
        self: object with get_default_config method.
        config: User-provided config.

    Returns:
        Config dictionary with defaults set.
    """
    default_config = self.get_default_config()
    if config is None:
        config = {}
    elif isinstance(config, str):
        config = read_config(config)
    elif not isinstance(config, dict):
        raise ConfigError("Config dictionary or filename expected, got {}".format(type(config)))

    # Check type.
    if CONFIG_TYPE in config:
        cls_name = type(self).__name__
        if cls_name != config[CONFIG_TYPE]:
            raise ConfigError("Type mismatch: expected {}, got {}".format(
                config[CONFIG_TYPE], cls_name))

    # Sample hyperparameters.
    _propagate_hyper_names(config)
    if CONFIG_HYPER in config:
        # Type of config[CONFIG_HYPER] is checked in _propagate_hyper_names.
        config = config.copy()
        for key, hopt in config[CONFIG_HYPER].items():
            # There can be unexpected hyperparameters for another implementation.
            # Skip them.
            if key not in default_config:
                continue
            config[key] = Hyper(**hopt).sample()

    # Merge configs.
    for key in config:
        if key in {CONFIG_TYPE, CONFIG_HYPER}:
            continue
        if key not in default_config:
            value = config[key]
            if isinstance(value, dict) and _is_hyper_only_config(value):
                # Subconfigs can contain hyper parameters for alternative configurations.
                pass
            else:
                raise ConfigError("Unknown parameter {}".format(key))
    new_config = OrderedDict()
    for key, value in default_config.items():
        new_config[key] = config.get(key, value)
    return new_config
