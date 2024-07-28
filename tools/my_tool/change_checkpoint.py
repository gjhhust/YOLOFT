import torch
import ultralytics.nn.modules.block as block

# Temporarily create a placeholder for VelocityNet_baseline3 to avoid loading errors
block.VelocityNet_baseline3 = block.MSTF

def rename_module_in_checkpoint(checkpoint_path, old_module_name, new_module_name, output_path):
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Check if the state_dict key exists in the checkpoint
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Create a new state_dict with renamed modules
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace(old_module_name, new_module_name)
        new_state_dict[new_key] = value

    # Update the checkpoint with the new state_dict
    if 'state_dict' in checkpoint:
        checkpoint['state_dict'] = new_state_dict
    else:
        checkpoint = new_state_dict

    # Save the modified checkpoint
    torch.save(checkpoint, output_path)
    print(f"Checkpoint saved to {output_path}")


if __name__ == "__main__":
    checkpoint_path = 'yoloft-L_old.pt'  # Replace with your checkpoint path
    old_module_name = 'VelocityNet_baseline3'      # Replace with the old module name
    new_module_name = 'MSTF'                       # Replace with the new module name
    output_path = 'yoloft-L.pt' # Replace with the desired output path
    rename_module_in_checkpoint(checkpoint_path, old_module_name, new_module_name, output_path)