import torch
import copy

class StreamTensorMemory(object):
    
    _instances = {} 
    def __new__(cls, *args, **kwargs):
        name = args[0] if args else None
        if name in cls._instances:
            # If an instance with the same name exists, the previous instance is returned.
            return cls._instances[name]
        else:
            instance = super().__new__(cls)
            instance.name = name
            cls._instances[name] = instance
            return instance
        
    def __init__(self, name):
         if not hasattr(self, 'initialized'):
            super().__init__()
            self.name = name
            self.bs = 1
            self.memory_list = [None]
            self.img_metas_memory = [None]
            self.initialized = True


    def update(self, memory, img_metas):
        assert len(memory) == len(img_metas)
        if len(memory) == self.bs:
            for i in range(self.bs):
                self.memory_list[i] = memory[i].clone().detach()
                self.img_metas_memory[i] = copy.deepcopy(img_metas[i])
        else: #batch size update
            self.bs = len(memory)
            self.memory_list = [None for i in range(self.bs)]
            self.img_metas_memory = [None for i in range(self.bs)]
            for i in range(self.bs):
                self.memory_list[i] = memory[i].clone().detach()
                self.img_metas_memory[i] = copy.deepcopy(img_metas[i])
        
    def reset_single(self, idx):
        self.memory_list[idx] = None
        self.img_metas_memory[idx] = None

    def reset_all(self):
        self.bs = 1
        self.memory_list = [None]
        self.img_metas_memory = [None]

    def get(self, img_metas):
        '''
        img_metas: list[img_metas]
        '''
        if len(img_metas) == self.bs:
            tensor_list = []
            img_metas_list = []
            is_first_frame_list = []
            
            for i in range(self.bs):
                
                if (self.img_metas_memory[i] == None)  or img_metas[i]["is_first"] :
                    is_first_frame = True
                else:
                    is_first_frame = False

                if is_first_frame:
                    self.reset_single(i)

                tensor_list.append(self.memory_list[i])
                img_metas_list.append(self.img_metas_memory[i])
                is_first_frame_list.append(is_first_frame)

            result = {
                'tensor': tensor_list,
                'img_metas': img_metas_list,
                'is_first_frame': is_first_frame_list,
            }
        else:
            self.bs = len(img_metas)
            self.memory_list = [None for i in range(self.bs)]
            self.img_metas_memory = [None for i in range(self.bs)]
            result = {
                'tensor': None,
                'img_metas': None,
                'is_first_frame': [True for i in range(self.bs)],
            }
            
        return result
    
class StreamTensorMutiMemory(object):
    
    _instances = {} 
    def __new__(cls, *args, **kwargs):
        name = args[0] if args else None
        if name in cls._instances:
            return cls._instances[name]
        else:
            instance = super().__new__(cls)
            instance.name = name
            cls._instances[name] = instance
            return instance
        
    def __init__(self, name, number_history):
         if not hasattr(self, 'initialized'):
            super().__init__()
            self.name = name
            self.bs = 1
            self.number_history = number_history
            self.memory_list = [None]
            self.img_metas_memory = [[None]]
            self.initialized = True


    def update(self, memory, img_metas):
        assert len(memory) == len(img_metas)
        if len(memory) == self.bs:
            for i in range(self.bs):
                if not img_metas[i]["is_first"]:
                    self.memory_list[i] = torch.stack([self.memory_list[i][1:],memory[i].unsqueeze(0).clone().detach()])
                    del self.img_metas_memory[i][0]
                    self.img_metas_memory[i].append(copy.deepcopy(img_metas[i]))
                else:
                    self.reset_single(i, memory[i], img_metas[i])
        else: #batch size update
            self.bs = len(memory)
            self.memory_list = [None for i in range(self.bs)]
            self.img_metas_memory = [None for i in range(self.bs)]
            for i in range(self.bs):
                self.reset_single(i, memory[i], img_metas[i])
                
    def reset_single(self, i, memory, img_meta):
        self.memory_list[i] = torch.zeros_like(self.memory_list[i])
        self.memory_list[i] = torch.stack([self.memory_list[i].expand(self.number_history-1, -1, -1, -1),
                                        memory.unsqueeze(0).clone().detach()])
        self.img_metas_memory[i] = [None for i in range(self.number_history)]
        self.img_metas_memory[i][-1] = copy.deepcopy(img_meta)
    
    def reset_all(self):
        self.bs = 1
        self.memory_list = [None]
        self.img_metas_memory = [None]

    def get(self, idx):
        '''
        img_metas: list[img_metas]
        '''
         
        return self.memory_list[idx].clone()


from collections import deque

class MutiFeatureBuffer(object):
    _instances = {}

    def __new__(cls, name, *args, **kwargs):
        # name = args[0] if args else None
        if name in cls._instances:
            # If an instance with the same name exists, the previous instance is returned.
            return cls._instances[name]
        else:
            instance = super().__new__(cls)
            instance.name = name
            cls._instances[name] = instance
            return instance

    def __init__(self, name, number_history, interval):
        if not hasattr(self, 'initialized'):
            super().__init__()
            self.name = name
            self.bs = 1
            self.interval = interval
            self.number_history = number_history
            self.maxlen = interval*number_history
            self.memory_list = {i: deque(maxlen=self.maxlen) for i in range(self.bs)}
            self.img_metas_memory = {i: deque(maxlen=self.maxlen) for i in range(self.bs)}
            self.initialized = True
            # 动态注册 get_all 方法
            if self.interval == 1:
                self.get_all = self._get_all_interval_one
            else:
                self.get_all = self._get_all_default


    def __reduce__(self):
        return (self.__class__, (self.name,self.number_history,self.interval))
    
    def update(self, memory, img_metas):
        assert len(memory) == len(img_metas)
        if len(memory) == self.bs and not self.is_empty():
            for i in range(self.bs):
                if not img_metas[i]["is_first"]:
                    self.memory_list[i].append(memory[i].clone())
                    self.img_metas_memory[i].append(copy.deepcopy(img_metas[i]))
                else:
                    self.reset_single(i, memory[i], img_metas[i])
        else:  # batch size update
            self.bs = len(memory)
            self.memory_list = {i: deque(maxlen=self.maxlen) for i in range(self.bs)}
            self.img_metas_memory = {i: deque(maxlen=self.maxlen) for i in range(self.bs)}
            for i in range(self.bs):
                self.reset_single(i, memory[i], img_metas[i])

    def reset_single(self, i, memory, img_meta):
        c, h, w = memory.shape
        zero_padding = torch.zeros([self.maxlen - 1, c, h, w], device=memory.device)
        self.memory_list[i].extend(zero_padding.detach())
        self.memory_list[i].append(memory.clone())
        
        self.img_metas_memory[i].extend([None] * (self.maxlen - 1))
        self.img_metas_memory[i].append(copy.deepcopy(img_meta))

    def reset_all(self):
        self.bs = 1
        self.memory_list = {i: deque(maxlen=self.maxlen) for i in range(self.bs)}
        self.img_metas_memory = {i: deque(maxlen=self.maxlen) for i in range(self.bs)}

    def get(self, batch_index, idx):
        return self.memory_list[batch_index][idx].clone()

    def _get_all_interval_one(self):
        """
        This method retrieves all elements from memory_list when interval is 1.
        """
        return torch.stack([torch.stack(list(mem)[-1::-self.interval], dim=0) for mem in self.memory_list.values()], dim=0).contiguous()

    def _get_all_default(self):
        """
        This method retrieves all elements from memory_list with default interval handling.
        """
        return torch.stack([torch.stack(list(mem)[-2::-self.interval], dim=0) for mem in self.memory_list.values()], dim=0).contiguous()

    # def get_all(self):
    #     return torch.stack([torch.stack(list(memory_list)[-2::-self.interval], dim=0) for memory_list in self.memory_list.values()], dim=0).contiguous()
    
    def is_empty(self):
    # Determine if the memory_list for each batch_index is empty.
        return all(not memory_list for memory_list in self.memory_list.values())

from .flow import  initialize_flow, coords_grid
import torch.nn.functional as F
class FeatureBuffer(object):
    _instances = {}

    def __new__(cls, name, *args, **kwargs):
        # name = args[0] if args else None
        if name in cls._instances:
            return cls._instances[name]
        else:
            instance = super().__new__(cls)
            instance.name = name
            cls._instances[name] = instance
            return instance
        
    def __init__(self, name, number_feature=3):
        super().__init__()
        if not hasattr(self, 'initialized'):
            self.name = name
            self.bs = 0
            self.memory_fmaps = None
            self.memory_bbox = None
            self.memory_score = None
            self.img_metas_memory =  [None for i in range(self.bs)]
            self.initialized = True
            self.number_feature = number_feature
            self.coords0 = None
            self.coords_orige = None
            self.ref_points_orige  = None
            self.spatial_shapes = None
            self.flow = None
            self.net = None

    def __reduce__(self):
        return (self.__class__, (self.name,self.number_feature))
    
    def convert_tensor_list(self, tensor_list):
        # separation gradient
        with torch.no_grad():
            l = len(tensor_list)
            assert self.number_feature == l, f"FeatureBuffer: number_feature is {self.number_feature} != truth fmaps number {l}"
            b, c, h, w = tensor_list[0].shape
            new_list = []
            for i in range(b):
                new_list.append([tensor_list[j][i].detach() for j in range(l)])

        return new_list, b
    
    def convert_list_tensor(self, list_): #[bs, 3, c, h, w]
        tensor_list = []
        for i in range(self.number_feature):
            tensor_list.append(torch.stack([x[i] for x in list_]))

        return tensor_list #[3, bs, c, h, w]
    
    def update_coords(self, flow):
        self.flow = flow.detach()

    def update_net(self, net):
        self.net = net.detach()

    def update_bbox(self, bbox, score):
        self.memory_bbox = bbox.detach()
        self.memory_score = score.detach()

    def zero_padding(self, memory_new, net_old, result_first_frame):
        bs, l, dim = memory_new.shape
        for i in range(self.bs): 
            if result_first_frame[i]: #If the current frame is the first, history is zeroed out, optical flow initialized
                self.net[i] = net_old[i]
                self.flow[i] = torch.zeros_like(self.coords0[i], device=self.coords0.device,dtype=self.coords0.dtype)
                self.memory_fmaps[i] = memory_new[i].clone()

    def update_memory(self, memory, net, img_metas, spatial_shapes):
        b, _, dim = memory.shape
        assert len(img_metas) == b

        if b == self.bs: # Store current memory and image information, return previous storage
            pass
        else:
            self.reset_all(b)

        # Initialized when coords_orige is None or inconsistent.
        if self.coords_orige is None or (self.coords_orige is not None and torch.equal(self.spatial_shapes,spatial_shapes)):
            # self.coords_orige, self.ref_points_orige = self.initialize_point(self.bs, spatial_shapes, memory[0][0].device)
            h, w = spatial_shapes[0]
            self.coords_orige, self.ref_points = self.initialize_point(self.bs, h, w, spatial_shapes, spatial_shapes.device)
            self.spatial_shapes = spatial_shapes.clone()
            self.coords0 = self.coords_orige.clone()
            self.flow = torch.zeros_like(self.coords0, device=self.coords_orige.device,dtype=self.coords_orige.dtype)

        result_first_frame = [img_metas[i]["is_first"] for i in range(b)]

        # buffer is empty, then it is also treated as the first frame
        if self.memory_fmaps is None:
            self.memory_fmaps = torch.zeros_like(memory, device=memory.device, dtype=memory.dtype)
            for i in range(self.bs):
                result_first_frame[i] = True

        if self.net is None:
            self.net = net.clone()

        # When there is no history frame, the corresponding position is replenished with zero.
        self.zero_padding(memory, net, result_first_frame)
        
        results_coords0 = self.coords0.clone()
        # results_coords1 = self.flow.clone() + results_coords0
        results_memory = self.memory_fmaps.clone()
        result_net = self.net.detach()

        #save
        self.net = net.clone()
        self.memory_fmaps = memory.detach()
        for i in range(self.bs):
            self.img_metas_memory[i] = copy.deepcopy(img_metas[i])

        # return [True for i in range(self.bs)], self.convert_list_tensor(results_memory), None
        return result_first_frame, results_memory, result_net, results_coords0, None, self.ref_points.detach(), [self.memory_bbox.clone(), self.memory_score.clone()] if self.memory_bbox is not None else None
    
    def reset_all(self, batch = 1):
        self.bs = batch
        self.memory_bbox = None
        self.memory_score = None
        self.memory_fmaps = None
        self.img_metas_memory =  [None for i in range(self.bs)]
        self.coords0 = None
        self.coords_orige = None
        self.spatial_shapes = None
        self.ref_points_orige  = None
        self.net = None

    # def initialize_point(self, bs, spatial_shapes, device):
    #     valid_ratios = torch.ones([bs, len(spatial_shapes), 2], device=device)
    #     ref_points, coords0 = get_reference_points(spatial_shapes, valid_ratios, device)
    #     return coords0, ref_points
    
    def initialize_point(self, bs, H, W, spatial_shapes, device):
        coords0, coords1 = initialize_flow(bs,  H,  W, device)
        valid_ratios = torch.ones([bs, len(spatial_shapes), 2], device=device)
        ref_points, _ = get_reference_points(spatial_shapes, valid_ratios, device)
        return coords0, ref_points

class FlowBuffer(object):
    _instances = {}

    def __new__(cls, name, *args, **kwargs):
        # name = args[0] if args else None
        if name in cls._instances:
            return cls._instances[name]
        else:
            instance = super().__new__(cls)
            instance.name = name
            cls._instances[name] = instance
            return instance
        
    def __init__(self, name, number_feature=3):
        super().__init__()
        if not hasattr(self, 'initialized'):
            self.name = name
            self.bs = 0
            self.memory_fmaps = None
            self.memory_bbox = None
            self.memory_score = None
            self.img_metas_memory =  [None for i in range(self.bs)]
            self.initialized = True
            self.number_feature = number_feature
            self.coords0 = None
            self.coords1 = None
            self.spatial_shapes = None
            self.net = None

    def __reduce__(self):
        return (self.__class__, (self.name,self.number_feature))
    
    def update_coords(self, coords1):
        self.coords1 = self.flatten([coords.detach() for coords in coords1])

    def update_net(self, nets):
        self.net = self.flatten([net.detach() for net in nets])

    def update_bbox(self, bbox, score):
        self.memory_bbox = bbox.detach()
        self.memory_score = score.detach()


    def update_memory(self, memory_flatten, img_metas, spatial_shapes, level_start_index):
        b, _, dim = memory_flatten.shape
        assert len(img_metas) == b

        if b == self.bs: # Store current memory and image information, return previous storage
            pass
        else:
            self.reset_all(b)

        # Initialized when coords_orige is None or inconsistent.
        if self.spatial_shapes is None or (not torch.equal(self.spatial_shapes,spatial_shapes)):
            # self.coords_orige, self.ref_points_orige = self.initialize_point(self.bs, spatial_shapes, memory[0][0].device)
            self.coords0 = self.initialize_point(self.bs, spatial_shapes)
            self.coords0 = self.flatten(self.coords0)
            self.spatial_shapes = spatial_shapes.clone()
            self.coords1 = self.coords0.clone()
            
            self.net = torch.tanh(memory_flatten).detach()
            self.memory_fmaps = memory_flatten.detach()
                            
        result_first_frame = [img_metas[i]["is_first"] for i in range(b)]

        if self.memory_fmaps is None:
            self.memory_fmaps = memory_flatten.detach()
            for i in range(self.bs):
                result_first_frame[i] = True

        if self.net is None:
            self.net = torch.tanh(memory_flatten).detach()

        for i in range(self.bs): 
            if result_first_frame[i]: # If current is first frame, history zero, stream initialization
                # print("Fist init flow")
                self.net[i] = torch.tanh(memory_flatten[i]).detach()
                self.coords1[i] = self.coords0[i]
                self.memory_fmaps[i] = memory_flatten[i].detach()

        results_coords0 = self.recover_src(self.coords0, spatial_shapes, level_start_index)
        results_coords1 = self.recover_src(self.coords1, spatial_shapes, level_start_index)
        results_memory = self.recover_src(self.memory_fmaps, spatial_shapes, level_start_index)
        result_net = self.recover_src(self.net, spatial_shapes, level_start_index)

        #save
        self.memory_fmaps = memory_flatten.detach()
        for i in range(self.bs):
            self.img_metas_memory[i] = copy.deepcopy(img_metas[i])

        return result_first_frame, results_memory, result_net, results_coords0, results_coords1, [self.memory_bbox.clone(), self.memory_score.clone()] if self.memory_bbox is not None else None
    
    def reset_all(self, batch = 1):
        self.bs = batch
        self.memory_bbox = None
        self.memory_score = None
        self.memory_fmaps = None
        self.img_metas_memory =  [None for i in range(self.bs)]
        self.coords1 = None
        self.coords0 = None
        self.spatial_shapes = None
        self.net = None

    # def initialize_point(self, bs, spatial_shapes, device):
    #     valid_ratios = torch.ones([bs, len(spatial_shapes), 2], device=device)
    #     ref_points, coords0 = get_reference_points(spatial_shapes, valid_ratios, device)
    #     return coords0, ref_points
    
    def initialize_point(self, bs, spatial_shapes):
        coords0 = []
        for lvl, shape in enumerate(spatial_shapes):
            h, w = shape
            coords0.append(coords_grid(bs, h, w, device=spatial_shapes.device))
        return coords0

    def recover_src(self, src_flatten, spatial_shapes, level_start_index):
        """
        Recovers the original list of src tensors from the flattened and processed tensors.

        Args:
            src_flatten: Flattened and concatenated source tensors (torch.Tensor).
            lvl_pos_embed_flatten: Flattened and concatenated positional embedding tensors (torch.Tensor).
            spatial_shapes: List of original spatial shapes (h, w) for each source level (torch.Tensor).
            level_start_index: Tensor indicating the starting index for each level in src_flatten (torch.Tensor).

        Returns:
            A list of source tensors with their original shapes.
        """
        srcs = []
        num_levels = level_start_index.size(0)

        bs, _, dim = src_flatten.shape
        for lvl in range(num_levels):
            start_index = level_start_index[lvl].item()
            end_index = level_start_index[lvl + 1].item() if lvl + 1 < num_levels else src_flatten.size(1)
            src = src_flatten[:, start_index:end_index].transpose(1, 2).reshape(bs, dim, *spatial_shapes[lvl])
            srcs.append(src)

        # return self.enc_output_norm(self.enc_output(srcs))
        return srcs


    def flatten(self, srcs, re_shape=False):
        # prepare input for encoder
        src_flatten = []
        spatial_shapes = []
        for lvl, src in enumerate(srcs):
            if re_shape:
                bs, c, h, w = src.shape
                spatial_shape = (h, w)
                spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            src_flatten.append(src)
        src_flatten = torch.cat(src_flatten, 1)
        if re_shape:
            spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
            level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        return src_flatten if not re_shape else (src_flatten, spatial_shapes, level_start_index)
    


def get_reference_points(spatial_shapes, valid_ratios, device):
    reference_points_list = []
    for lvl, (H_, W_) in enumerate(spatial_shapes):

        ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                        torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
        if lvl==0:
            coords = torch.stack([ref_y/H_, ref_x/W_], dim=0).float()
            coords = coords[None].repeat(valid_ratios.shape[0], 1, 1, 1)
        ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
        ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
        ref = torch.stack((ref_x, ref_y), -1)
        reference_points_list.append(ref)
    reference_points = torch.cat(reference_points_list, 1)
    # reference_points = reference_points[:, :, None] * valid_ratios[:, None]
    reference_points = reference_points.unsqueeze(-2).repeat(1,1,len(spatial_shapes),1)
    #len(spatial_shapes) normalized reference points for each feature point -> [bs, H/8*W/8+H/16*W/16+H/32*W/32, len(spatial_shapes), 2]
    return reference_points, coords

def from_coords_refpoint(coords, spatial_shapes): 
    '''
    coords are the optical flow coordinates of the topmost layer, downsampled to get the ref points of the other layers (normalized 0-1)
    '''
    bs = coords.shape[0]
    reference_points_list = []
    # batch_size, _, height, width = coords.shape
    h,w = spatial_shapes[0]

    assert h==w

    for lvl, (H_, W_) in enumerate(spatial_shapes):
        if lvl == 0:
            ref_coords = coords.clone()
        else:
            ref_coords = F.interpolate(coords, size=(H_, W_), mode='bilinear', align_corners=True) #bs 2 H W

        ref_coords[:,0,:,:] = ref_coords[:,0,:,:] / h
        ref_coords[:,1,:,:] = ref_coords[:,1,:,:] / w

        ref_coords = ref_coords.reshape(bs, 2,-1).transpose(-1, -2)

        reference_points_list.append(ref_coords)

    reference_points = torch.cat(reference_points_list, -2)
    reference_points = reference_points.unsqueeze(-2).repeat(1,1,len(spatial_shapes),1)
    return reference_points


def from_refpoint_coords(reference_points, spatial_shapes):
    '''
    Multi-layer reference point extraction of the reference point with the largest resolution (image size)
    '''
    reference_points = reference_points.mean(-2)
    bs,_,dim = reference_points.shape
    H,W = spatial_shapes[0]
    return reference_points[:,:H*W,:].transpose(-1, -2).reshape(bs, dim, H, W)



class StreamBuffer(object):
    _instances = {}

    def __new__(cls, name, *args, **kwargs):
        # name = args[0] if args else None
        if name in cls._instances:
            return cls._instances[name]
        else:
            instance = super().__new__(cls)
            instance.name = name
            cls._instances[name] = instance
            return instance
        
    def __init__(self, name, number_feature=3):
        super().__init__()
        if not hasattr(self, 'initialized'):
            self.name = name
            self.bs = 0
            self.memory_fmaps = None
            self.img_metas_memory =  [None for i in range(self.bs)]
            self.initialized = True
            self.number_feature = number_feature


    def __reduce__(self):
        return (self.__class__, (self.name,self.number_feature))
    
    def get_spatial_shapes(self, srcs):
        spatial_shapes = []
        for lvl, src in enumerate(srcs):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
        return torch.as_tensor(spatial_shapes, dtype=torch.long, device=srcs[0].device)

    def update_memory(self, memory_now, img_metas):
        b, dim, h, w = memory_now[0].shape
        spatial_shapes = self.get_spatial_shapes(memory_now)
        
        assert len(img_metas) == b

        if b == self.bs: # Store current memory and image information, return previous storage
            pass
        else:
            self.reset_all(b)

        result_first_frame = [img_metas[i]["is_first"] for i in range(b)]
        
        # must init
        if torch.is_tensor(self.spatial_shapes) and (self.spatial_shapes.device != spatial_shapes.device):
            self.spatial_shapes = None
        if self.spatial_shapes is None or (not torch.equal(self.spatial_shapes,spatial_shapes)) or self.memory_fmaps is None:
            self.spatial_shapes = spatial_shapes.clone()
            with torch.no_grad():
                self.memory_fmaps = [f.detach().clone() for f in memory_now]
            
            for i in range(self.bs):
                result_first_frame[i] = True
                
        # init video
        with torch.no_grad():
            for i in range(self.bs): 
                if result_first_frame[i]: # If current is first frame, history zero, stream initialization
                    # print("Fist init flow")
                    # update every level
                    for f in range(len(memory_now)):
                        self.memory_fmaps[f][i] = memory_now[f][i].detach().clone()

        #save
        results_memory = [f.clone() for f in self.memory_fmaps]
        with torch.no_grad():
            self.memory_fmaps = [f.detach().clone() for f in memory_now]
        for i in range(self.bs):
            self.img_metas_memory[i] = copy.deepcopy(img_metas[i])

        return result_first_frame, results_memory
    
    def reset_all(self, batch = 1):
        self.bs = batch
        self.memory_fmaps = None
        self.img_metas_memory =  [None for i in range(self.bs)]
        self.spatial_shapes = None