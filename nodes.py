import comfy
import folder_paths
import torch
import comfy.ops
from .utils import save_imatrix
import os

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
DATE_DIR = os.path.join(CURRENT_DIR, "imatrix_data")

class ImatrixOps(comfy.ops.manual_cast):
    class Linear(comfy.ops.manual_cast.Linear):
        def __init__(self, in_features, out_features, *args, **kwargs):
            super().__init__(in_features, out_features, *args, **kwargs)
            self.imatrix = torch.ones(in_features)
            self.num_counts = 0

        def forward(self, x, *args, **kwargs):
            self.num_counts += 1
            imatrix = x.detach().clone().float().pow(2).mean(dim=list(range(len(x.shape)))[:-1]).cpu()
            self.imatrix = imatrix / self.num_counts + self.imatrix * (self.num_counts - 1) / self.num_counts
            return super().forward(x, *args, **kwargs)
    
    class Conv2d(comfy.ops.manual_cast.Conv2d):
        def __init__(self, in_channels, out_channels, kernel_size, *args, **kwargs):
            super().__init__(in_channels, out_channels, kernel_size, *args, **kwargs)
            self.imatrix = torch.ones(in_channels * self.kernel_size[0] * self.kernel_size[1])
            self.num_counts = 0

        def forward(self, x, *args, **kwargs):
            self.num_counts += 1
            imatrix = x.detach().clone().float().pow(2).mean(dim=(0, 2, 3)).cpu().repeat(self.kernel_size[0] * self.kernel_size[1])
            self.imatrix = imatrix / self.num_counts + self.imatrix * (self.num_counts - 1) / self.num_counts
            return super().forward(x, *args, **kwargs)

class ImatrixUNETLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "unet_name": (folder_paths.get_filename_list("diffusion_models"), ),
                "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],)
            }
        }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"

    CATEGORY = "imatrix"

    def load_unet(self, unet_name, weight_dtype="default"):
        model_options = {}
        if weight_dtype == "fp8_e4m3fn":
            model_options["dtype"] = torch.float8_e4m3fn
        elif weight_dtype == "fp8_e4m3fn_fast":
            model_options["dtype"] = torch.float8_e4m3fn
            model_options["fp8_optimizations"] = True
        elif weight_dtype == "fp8_e5m2":
            model_options["dtype"] = torch.float8_e5m2
        
        model_options["custom_operations"] = ImatrixOps()
        unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
        model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)
        return (model,)
    
class SaveImatrix:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "file_name": ("STRING", {"default": "imatrix"}),
            },
            "optional": {
                "image_not_used": ("IMAGE", ),
            }
        }
    
    RETURN_TYPES = ()
    FUNCTION = "save_imatrix"

    CATEGORY = "imatrix"
    OUTPUT_NODE = True

    def save_imatrix(self, model, file_name, image_not_used=None):
        imatrix_data = {}
        for name, module in model.model.diffusion_model.named_modules():
            if hasattr(module, "imatrix") and module.imatrix is not None:
                imatrix_data[name + ".weight"] = module.imatrix.float().cpu().numpy().tolist()
        
        imatrix_file = os.path.join(DATE_DIR, f"{file_name}.dat")
        save_imatrix(imatrix_file, imatrix_data)
        
        print(f"Saved importance matrix to {imatrix_file}")
        return {}