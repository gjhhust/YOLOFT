import torch

def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    return checkpoint

def save_checkpoint(checkpoint, checkpoint_path):
    torch.save(checkpoint, checkpoint_path)

def group_by_module(state_dict):
    module_dict = {}
    for key, value in state_dict.items():
        module_name = '.'.join(key.split('.')[:2])
        if module_name not in module_dict:
            module_dict[module_name] = {}
        module_dict[module_name][key] = value
    return module_dict

def compare_submodules_and_update(new_module, old_module, transferred_state_dict):
    for k in list(old_module.keys()):
        if "convs0." in k:
            del old_module[k]
    for k, v in new_module.items():
        submodule_name = '.'.join(k.split('.')[2:])
        if submodule_name in old_module and old_module[submodule_name].shape == v.shape:
            transferred_state_dict[k] = old_module[submodule_name]
        else:
            print(f"Shape mismatch for {k}: {v.shape} != {old_module.get(submodule_name, 'not found').shape}")

def manual_match_and_transfer(new_modules, old_modules, match_dict, transferred_state_dict):
    for new_module_name, old_module_name in match_dict.items():
        if new_module_name in new_modules and old_module_name in old_modules:
            new_module = new_modules[new_module_name]
            old_module = {k[len(old_module_name)+1:]: v for k, v in old_modules[old_module_name].items()}
            compare_submodules_and_update(new_module, old_module, transferred_state_dict)
        else:
            print(f"Module pair {new_module_name} and {old_module_name} not found in respective checkpoints")

def automatic_match_and_transfer(new_modules, old_modules, transferred_state_dict, match_dict):
    remaining_new_modules = {k: v for k, v in new_modules.items() if k not in match_dict}
    remaining_old_modules = {k: v for k, v in old_modules.items() if k not in match_dict}
    
    for new_module_name, new_module in remaining_new_modules.items():
        for old_module_name, old_module in remaining_old_modules.items():
            if compare_shapes_and_update(new_module, old_module, transferred_state_dict):
                del remaining_old_modules[old_module_name]
                break
            else:
                print(f"No matching module for {new_module_name}")
                for k, v in new_module.items():
                    print(f"    {k}: {v.shape}")

def compare_shapes_and_update(new_module, old_module, transferred_state_dict):
    if len(new_module) != len(old_module):
        return False
    for (new_k,new_v),(old_k,old_v) in zip(new_module.items(),old_module.items()):
        new_submodule_name = '.'.join(new_k.split('.')[2:])
        old_submodule_name = '.'.join(old_k.split('.')[2:])
        if new_submodule_name == old_submodule_name and new_v.shape == old_v.shape:
            transferred_state_dict[new_k] = old_v
        else:
            print(f"error: {new_submodule_name}:{new_v.shape}, {new_submodule_name}:{old_v.shape}")
            return False
    return True

import re

def extract_module_number(key):
    match = re.search(r'model\.(\d+)\.', key)
    if match:
        return int(match.group(1))
    return None

if __name__ == "__main__":
    old_checkpoint_path = 'YoloftS_new_convnoany_32.6.pt'  # Replace with your checkpoint path
    new_checkpoint_path = 'yoloft/train5/weights/best.pt'
    output_checkpoint_path = 'new_YoloftS_new_convnoany_32.6_2.pt'

    # Manually defined matching dictionary
    match_dict = {
        'model.12': 'model.12',
        # Add more pairs as needed
    }

    # Load old and new checkpoints
    old_checkpoint = load_checkpoint(old_checkpoint_path)
    new_checkpoint = load_checkpoint(new_checkpoint_path)

    # Get state dicts and convert them to float
    old_state_dict = old_checkpoint['model'].float().state_dict()
    new_state_dict = new_checkpoint['model'].float().state_dict()

    cnt = 0
    transferred_state_old_dict = {}
    for k in list(old_state_dict.keys()):
        if "convs0." in k:
            del old_state_dict[k]
            continue
        module_number = extract_module_number(k)
        if module_number is not None:
            adjusted_number = module_number - 2 if module_number > 12 else module_number
            module_name = f'model.{adjusted_number}'
            new_key = k.replace(f'model.{module_number}', module_name)
            if old_state_dict[k].shape == new_state_dict[new_key].shape:
                transferred_state_old_dict[new_key] = old_state_dict[k]
            else:
                print("error")

        # Load the new weights into the old model's state dict
    res = old_checkpoint['model'].load_state_dict(transferred_state_old_dict, strict=False)

    # Optionally save the updated checkpoint
    torch.save(old_checkpoint, 'updated_checkpoint.pt')

    # Update new checkpoint with transferred weights
    new_checkpoint['model'].load_state_dict(transferred_state_old_dict)

    # Save the new checkpoint
    save_checkpoint(new_checkpoint, output_checkpoint_path)
    print(f"New checkpoint saved at {output_checkpoint_path}")
