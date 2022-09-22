from types import SimpleNamespace

import toml
import yaml

if __name__ == '__main__':
    with open('test.yaml') as yaml_file:
        config_yaml = yaml.load(yaml_file, Loader=yaml.Loader)

    with open('test.toml') as toml_file:
        config_toml = toml.load(toml_file)

    config_toml = SimpleNamespace(**config_toml)
