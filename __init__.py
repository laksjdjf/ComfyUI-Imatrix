from .nodes import ImatrixUNETLoader, SaveImatrix
NODE_CLASS_MAPPINGS = {
    "ImatrixUNETLoader": ImatrixUNETLoader,
    "SaveImatrix": SaveImatrix,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImatrixUNETLoader": "Imatrix UNet Loader",
    "SaveImatrix": "Save Imatrix",
}

__all__ = [NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS]