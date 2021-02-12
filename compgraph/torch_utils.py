import torch

def get_batch_from_cache(inputs, cache, cache_device, output_device_dict):
    values = [cache.get(key, None) for key in inputs.keys]
    if any(value is None for value in values):
        return None
    one_key, one_value = inputs.keys[0], values[0]
    if isinstance(one_value, torch.Tensor):
        # stack all the values together into one tensor
        stack_dim = 0 if one_value.dim() == 0 else inputs.batch_dim
        result = torch.stack(values, dim=stack_dim)
        if cache_device is not None:
            output_device = output_device_dict[one_key]
            if output_device != cache_device:
                return result.to(output_device)
    else:
        return values

def save_batch_to_cache(inputs, result, cache, cache_device, output_device_dict):
    result_for_cache = result
    if cache_device is not None and isinstance(result, torch.Tensor):
        if result.device != cache_device:
            result_for_cache = result.to(cache_device)
        output_device_dict.update((key, result.device) for key in inputs.keys)
    if inputs.batch_dim == 0 or isinstance(result_for_cache, torch.Tensor) \
            and result_for_cache.dim() == 1:
        cache.update((key, value) for key, value in
                     zip(inputs.keys, result_for_cache))

    elif isinstance(result_for_cache, torch.Tensor):
        # print("results_for_cache.shape", result_for_cache.shape)
        split = result_for_cache.split(1, dim=inputs.batch_dim)
        # print("split[0].shape", split[0].shape)
        cache.update((key, value.squeeze(inputs.batch_dim))
                     for key, value in zip(inputs.keys, split))
    else:
        raise RuntimeError(f"Does not support type {type(result_for_cache)} "
                           f"during computation for batch_dim={inputs.batch_dim}")