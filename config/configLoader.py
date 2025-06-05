import yaml

def load_config(config_path="config/config.yaml"):
    """
    Load the configuration from a YAML file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict: The parsed configuration as a dictionary.
    """
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
