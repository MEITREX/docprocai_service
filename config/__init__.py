import yaml

with open("./../config.yaml") as file:
    current = yaml.load(file, Loader=yaml.FullLoader)
