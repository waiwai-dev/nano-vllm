import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    loaded = set()
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for source_name in f.keys():
                weight_name = getattr(model, "checkpoint_name", lambda name: name)(source_name)
                if weight_name is None:
                    continue
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(source_name), shard_id)
                        loaded.add(param_name)
                        break
                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(source_name))
                    loaded.add(weight_name)
    # Fail before generation rather than running with uninitialized parameters.
    loaded_storage = {model.get_parameter(name).data_ptr() for name in loaded}
    missing = [name for name, param in model.named_parameters() if param.data_ptr() not in loaded_storage]
    if missing:
        raise ValueError(f"Missing checkpoint weights: {missing}")
