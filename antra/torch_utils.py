import torch
import numpy as np

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
        return result
    elif isinstance(one_value, dict):
        stack_dim = 0 if one_value[list(one_value.keys())[0]].dim() == 0 else inputs.batch_dim
        # let us resemble batched outputs from dicts.
        batched_keys = [k for k in one_value.keys()]
        sliced_values = [list(v.values()) for v in values]
        batched_values = []
        for v in zip(*sliced_values):
            # depending on the type, we decide how to consolidate them.
            if isinstance(v[0], torch.Tensor):
                cat_v = torch.cat(v, dim=stack_dim)
            elif isinstance(v[0], list):
                cat_v = []
                for ele in v:
                    cat_v.append(ele[0])
            else:
                cat_v = []
                for ele in v:
                    cat_v.append(ele[0])
                cat_v = np.array(cat_v)
            batched_values += [cat_v]
        if cache_device is not None:
            output_device = output_device_dict[one_key]
            if output_device != cache_device:
                for v in batched_values:
                    if isinstance(v, torch.Tensor):
                        v.to(output_device)
        return dict(zip(batched_keys, batched_values))
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
        # if the result is in a dictionary format
        if isinstance(result_for_cache, dict):
            # we need to broadcast the dict keys.
            batched_keys = [k for k in result_for_cache.keys()]
            batched_values = []
            for v in result_for_cache.values():
                print(v)
                if isinstance(v, torch.Tensor):
                    batched_values += [v.split(1, dim=inputs.batch_dim)]
                elif isinstance(v, list):
                    # we assume this then is a list?
                    v_splits = [[v_slice] for v_slice in v]
                    batched_values += [v_splits]
                else:
                    # it is probably a numpy array?
                    v_splits = [np.array([v_slice]) for v_slice in v]
                    batched_values += [v_splits]

            broadcast_result_for_cache = []
            for value in zip(*batched_values):
                broadcast_result_for_cache += [dict(zip(batched_keys, value))]
            cache.update((key, value) for key, value in
                         zip(inputs.keys, broadcast_result_for_cache))
        else:
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


def deserialize(x: tuple):
    return torch.tensor(x)