import comfy
import folder_paths
import torch
import comfy.ops
from .utils import save_imatrix
import os

import gguf
TORCH_COMPATIBLE_QTYPES = (None, gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16)
def is_torch_compatible(tensor):
    return tensor is None or getattr(tensor, "tensor_type", None) in TORCH_COMPATIBLE_QTYPES

def is_quantized(tensor):
    return not is_torch_compatible(tensor)

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
            self.imatrix = torch.ones(in_channels)
            self.num_counts = 0

        def forward(self, x, *args, **kwargs):
            self.num_counts += 1
            imatrix = x.detach().clone().float().pow(2).mean(dim=(0, 2, 3)).cpu()
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
                tensor = module.imatrix.float().cpu()

                if isinstance(module, comfy.ops.manual_cast.Conv2d):
                    tensor = tensor.repeat_interleave(module.kernel_size[0] * module.kernel_size[1])
                multiplier = 1
                input_dims = tensor.shape[0]
                while input_dims % 256 != 0:
                    input_dims *= 2
                    multiplier *= 2
                tensor = tensor.repeat(multiplier)
                imatrix_data[name + ".weight"] = tensor.numpy().tolist()
        
        imatrix_file = os.path.join(DATE_DIR, f"{file_name}.dat")
        save_imatrix(imatrix_file, imatrix_data)
        
        print(f"Saved importance matrix to {imatrix_file}")
        return {}

class LoRAdiff:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "gguf_model": ("MODEL", ),
                "rank": ("INT", {
                    "default": 16, 
                    "min": 1, #Minimum value
                    "max": 320, #Maximum value
                    "step": 1, #Slider's step
                    "display": "number" # Cosmetic only: display as "number" or "slider"
                }),
                "device": (["cuda", "cpu"], ),
                "dtype": (["float32", "float16", "bfloat16"], ),
                "file_name": ("STRING", {"multiline": False, "default": "diff"}),
                "extension": (["safetensors"], ),
            }
        }
    
    RETURN_TYPES = ()
    FUNCTION = "lora_diff"

    CATEGORY = "imatrix"
    OUTPUT_NODE = True

    def lora_diff(self, model, gguf_model, rank, device="cuda", dtype="float32", file_name="merdiffged", extension="safetensors"):
        dtype = torch.float32 if dtype == "float32" else torch.float16 if dtype == "float16" else torch.bfloat16
        state_dict = {}
        for (name_org, module_org), (name_gguf, module_gguf) in zip(
            model.model.diffusion_model.named_modules(), 
            gguf_model.model.diffusion_model.named_modules()
        ):
            if hasattr(module_org, "weight") and is_quantized(module_gguf.weight):
                weight_org = module_org.weight.data.detach().clone().to(device=device, dtype=torch.float32)
                weight_gguf = module_gguf.get_weight(module_gguf.weight.to("cuda"), dtype=torch.float32).detach().clone().to(device=device)

                diff = weight_org - weight_gguf
                org_shape = diff.shape

                U, S, Vh = torch.linalg.svd(diff.flatten(1))

                U = U[:, :rank]
                S = S[:rank]
                U = U @ torch.diag(S)
                Vh = Vh[:rank, :]
                dist = torch.cat([U.flatten(), Vh.flatten()])
                hi_val = torch.quantile(dist, 0.99)
                low_val = -hi_val

                U = U.clamp(low_val, hi_val)
                Vh = Vh.clamp(low_val, hi_val)

                if len(org_shape) == 4:
                    U = U.reshape(org_shape[0], rank, 1, 1)
                    Vh = Vh.reshape(rank, org_shape[1], org_shape[2], org_shape[3])
                
                prefix_key = "lora_unet_" + name_org.replace(".", "_")
                state_dict[prefix_key + ".lora_up.weight"] = U.to(device=device, dtype=dtype)
                state_dict[prefix_key + ".lora_down.weight"] = Vh.to(device=device, dtype=dtype)
                state_dict[prefix_key + ".alpha"] = torch.tensor(rank, device=device, dtype=dtype)

                print(f"key: {prefix_key}, rank: {rank}, shape: {org_shape}, up_shape: {U.shape}, down_shape: {Vh.shape}")
        
        save_path = os.path.join(folder_paths.folder_names_and_paths["loras"][0][0], file_name + "." + extension)
        comfy.utils.save_torch_file(state_dict, save_path)

        return {}

                

