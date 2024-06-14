import torch
old_weight = torch.load("/data1/jiahaoguo/ultralytics/runs/detect/train142/weights/best.pt", map_location='cpu')
yolo_weight = torch.load("/data1/jiahaoguo/ultralytics/pth/movedet_mlp2_new.pt", map_location='cpu')

# old_keys = []
# for key,value in old_weight.items():
#     old_keys.append(key)

old_keys = []
for key,value in old_weight["model"].state_dict().items():
    old_keys.append(key)
old_keys = old_keys[252:423]

yolo_keys = []
for key,value in yolo_weight["model"].state_dict().items():
    yolo_keys.append(key)

map_dict = {}
for i in range(len(old_keys)-1):
    print(f"{i}:  {old_keys[i]} \t\t\t\t\t {yolo_keys[i]}")
    map_dict[yolo_keys[i]] = old_weight["model"].state_dict()[old_keys[i]]

for key ,value in map_dict.items():
    yolo_weight["model"].state_dict()[key] = value

torch.save(yolo_weight,'/data1/jiahaoguo/ultralytics/pth/movedet_mlp2.pt')