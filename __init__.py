from .nodes import ImatrixUNETLoader, SaveImatrix, LoRAdiff
NODE_CLASS_MAPPINGS = {
    "ImatrixUNETLoader": ImatrixUNETLoader,
    "SaveImatrix": SaveImatrix,
    "LoRAdiff": LoRAdiff,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImatrixUNETLoader": "Imatrix UNet Loader",
    "SaveImatrix": "Save Imatrix",
    "LoRAdiff": "LoRA diff",
    
}

__all__ = [NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS]