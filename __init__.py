# __init__.py

from .kontext_api import ModelScopeKontextAPI

# A dictionary that contains all nodes you want to export with their names
# The names should be descriptive and searchable in the frontend
NODE_CLASS_MAPPINGS = {
    "ModelScopeKontextAPI": ModelScopeKontextAPI
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelScopeKontextAPI": "ModelScope Kontext API"
}