import torch
import copy


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

    def reset_all(self, batch=1):
        if self.memory_fmaps is not None:
            for f in self.memory_fmaps:
                del f
        self.memory_fmaps = None

        self.img_metas_memory = [None for _ in range(batch)]
        self.spatial_shapes = None
        torch.cuda.empty_cache()


    def update_memory(self, memory_now, img_metas):
        b, dim, h, w = memory_now[0].shape
        spatial_shapes = self.get_spatial_shapes(memory_now)
        assert len(img_metas) == b

        if b != self.bs:  # Reset if batch size changes
            self.reset_all(b)

        result_first_frame = [img_metas[i]["is_first"] for i in range(b)]

        if self.spatial_shapes is None or not torch.equal(self.spatial_shapes, spatial_shapes):
            self.spatial_shapes = spatial_shapes
            if self.memory_fmaps is not None:
                for f in self.memory_fmaps:
                    del f
            self.memory_fmaps = [f.detach() for f in memory_now]
            for i in range(self.bs):
                result_first_frame[i] = True

        with torch.no_grad():
            for i in range(self.bs):
                if result_first_frame[i]:
                    for f in range(len(memory_now)):
                        self.memory_fmaps[f][i] = memory_now[f][i].detach()

        results_memory = [f.detach().clone() for f in self.memory_fmaps]
        with torch.no_grad():
            self.memory_fmaps = [f.detach().clone() for f in memory_now]
        for i in range(self.bs):
            self.img_metas_memory[i] = img_metas[i].copy()

        return result_first_frame, results_memory


class StreamBuffer_onnx(object):
    def __init__(self, number_feature=3):
        super().__init__()
        self.bs = 0
        self.number_feature = number_feature
        self.spatial_shape = None

    def update_memory(self, memory_last, img_metas, spatial_shape):
        spatial_shape = torch.as_tensor(spatial_shape, device=memory_last[0].device)
        
        memory_fmaps = [f.clone() for f in memory_last]

        b, dim, h, w = memory_last[0].shape
        assert len(img_metas) == b
        self.bs = b
        result_first_frame = [img_metas[i]["is_first"] for i in range(b)]
        
        if self.spatial_shape is None:
            self.spatial_shape = spatial_shape
            
        if not torch.equal(self.spatial_shape, spatial_shape):
            self.spatial_shape = spatial_shape
            return result_first_frame, [None, None, None]
            
        for i in range(self.bs):
            if result_first_frame[i]:
                for f in range(len(memory_last)):
                    memory_fmaps[f][i] = torch.zeros_like(memory_last[f][i], device=memory_last[f][i].device)

        return result_first_frame, memory_fmaps
