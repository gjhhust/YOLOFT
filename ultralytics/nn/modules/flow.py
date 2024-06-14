import torch
import torch.nn as nn
import torch.nn.functional as F
from .block import flow_conv
try:
    # import alt_cuda_corr
    import alt_cuda_sparse_corr as alt_cuda_corr
except:
    # alt_cuda_corr is not compiled
    pass

def warp_feature(x, coords):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    coords: [B, 2, H, W] flow+coords0 denotes the destination to which each pixel of x flows
    """
    B, C, H, W = x.size()
    vgrid = coords.clone()
    vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone()/max(W-1,1)-1.0 
    #Taking out the dimension of optical flow v, it turns out that the range is 0 to W-1, then dividing by W-1, the range is 0 to 1, then multiplying by 2, the range is 0 to 2, then -1, the range is -1 to 1
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone()/max(H-1,1)-1.0 #Taking out the dimension of optical flow u, as above

    vgrid = vgrid.permute(0,2,3,1)#from B,2,H,W -> B,H,W,2
    output = nn.functional.grid_sample(x, vgrid,align_corners=True)
    mask = torch.ones_like(x)
    mask = nn.functional.grid_sample(mask, vgrid,align_corners=True)

    mask[mask<0.9999] = 0
    mask[mask>0] = 1

    return output*mask

def coords_grid(batch, ht, wd, device):
    coords = torch.meshgrid(torch.arange(ht, device=device), torch.arange(wd, device=device), indexing="ij")
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def upflow8(flow, mode='bilinear'):
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return  8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)


def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img

def initialize_flow(b, H, W, device):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        coords0 = coords_grid(b, H, W, device=device)
        coords1 = coords_grid(b, H, W, device=device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1
        
class CorrBlock:
    """fmap1 [list], num_levels, 0-num_levels with Larger resolution change"""
    def __init__(self, fmap1, fmaps2, num_levels=3, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        assert len(fmaps2) == num_levels, f"fmap1 and fmap2 must len euqal num_levels"

        # all pairs correlation 
        # batch, dim, h1, w1 = fmap1.shape
        # self.fmap1 = fmap1.detach()
        # self.fmaps2 = [fmap2.detach() for fmap2 in fmaps2] # batch, dim, h2, w2

        for i in range(self.num_levels):
            corr = CorrBlock.corr(fmap1, fmaps2[i])
            batch, h1, w1, dim, h2, w2 = corr.shape
            corr = corr.reshape(batch*h1*w1, dim, h2, w2)
            self.corr_pyramid.append(corr)


    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            dx = torch.linspace(-r, r, 2*r+1, device=coords.device)
            dy = torch.linspace(-r, r, 2*r+1, device=coords.device)
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1)

            centroid_lvl = coords.reshape(batch*h1*w1, 1, 1, 2) / 2**i #Get the coordinates of the current pyramid zoom level, each value in coords represents the displacement of the pixel at xy (2, which represents the xy direction displacement)
            delta_lvl = delta.view(1, 2*r+1, 2*r+1, 2)
            coords_lvl = centroid_lvl + delta_lvl # Broadcast to each pixel point, generating a sample point for each pixel point (batch*h1*w1, 2*r+1, 2*r+1, 2)

            corr = self.corr_pyramid[i]
            corr = bilinear_sampler(corr, coords_lvl)

            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()
    
    def bilinear_sampler_cil(self, fmap1, fmap2, coords, mode='bilinear', mask=False):
        '''
        coords_lvl denotes the correlation sampling of fmap1 to fmap2
        '''
        """ Wrapper for grid_sample, uses pixel coordinates """
        H, W = img.shape[-2:]
        xgrid, ygrid = coords.split([1,1], dim=-1)
        xgrid = 2*xgrid/(W-1) - 1
        ygrid = 2*ygrid/(H-1) - 1

        grid = torch.cat([xgrid, ygrid], dim=-1)
        img = F.grid_sample(img, grid, align_corners=True)

        if mask:
            mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
            return img, mask.float()

        return img

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        batch, dim, ht2, wd2 = fmap2.shape
        fmap1 = fmap1.view(batch, dim, ht*wd)
        fmap2 = fmap2.view(batch, dim, ht2*wd2) 
        
        corr = torch.matmul(fmap1.transpose(1,2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht2, wd2)
        return corr  / torch.sqrt(torch.tensor(dim).float())   
    
class AlternateCorrBlock:
    def __init__(self, fmap1, fmaps2, num_levels=4, radius=4, stride=[1,1,1]):
        self.num_levels = num_levels
        self.radius = radius
        self.stride = stride
        
        self.pyramid = []
        for i in range(self.num_levels):
            self.pyramid.append((fmap1, fmaps2[i]))

    def __call__(self, coords):
        coords = coords.permute(0, 2, 3, 1)
        B, H, W, _ = coords.shape
        dim = self.pyramid[0][0].shape[1]

        corr_list = []
        for i in range(self.num_levels):
            fmap1_i = self.pyramid[0][0].permute(0, 2, 3, 1).contiguous()
            fmap2_i = self.pyramid[i][1].permute(0, 2, 3, 1).contiguous()

            coords_i = (coords / 2**i).reshape(B, 1, H, W, 2).contiguous()

            # with torch.cuda.amp.autocast(enabled=False):
            corr, = alt_cuda_corr.forward(fmap1_i.float(), fmap2_i.float(), coords_i.float(), self.radius, self.stride[i])
            # nan_tensor = torch.isnan(corr)
            # if nan_tensor.any():
            #     corr[nan_tensor] = 0.0
            corr_list.append(corr.squeeze(1))

        corr = torch.stack(corr_list, dim=1)
        corr = corr.reshape(B, -1, H, W)
        return corr / torch.sqrt(torch.tensor(dim).float())

class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead, self).__init__()
        self.conv1 = flow_conv(input_dim, hidden_dim, 3, p=1)
        self.conv2 = flow_conv(hidden_dim, 2, 3, p=1)

    def forward(self, x):
        return self.conv2(self.conv1(x))

class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))

        h = (1-z) * h + z * q
        return h

class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))

        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))


    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))        
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))       
        h = (1-z) * h + z * q

        return h

class SmallMotionEncoder(nn.Module):
    def __init__(self, input_dim=128, output_idm=80):
        super(SmallMotionEncoder, self).__init__()
        self.convc1 = flow_conv(input_dim, 96, 1, p=0)
        self.convf1 = flow_conv(2, 64, 7, p=3)
        self.convf2 = flow_conv(64, 32, 3, p=1)
        self.conv = flow_conv(128, output_idm, 3, p=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)

class BasicMotionEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BasicMotionEncoder, self).__init__()
        self.convc1 = nn.Conv2d(input_dim, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(64+192, output_dim, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)

class SmallUpdateBlock(nn.Module):
    def __init__(self, hidden_dim=64, input_dim=64, corr_levels=3, corr_radius=4):
        super(SmallUpdateBlock, self).__init__()
        cor_planes = corr_levels * (2*corr_radius + 1)**2
        # self.encoder = SmallMotionEncoder(input_dim+hidden_dim, input_dim-2) #outdim = input_dim
        self.encoder = SmallMotionEncoder(cor_planes, hidden_dim-2) #outdim = hidden_dim
        self.gru = ConvGRU(hidden_dim=hidden_dim, input_dim=input_dim+hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=128)

    def forward(self, net, inp, corr, flow):
        motion_features = self.encoder(flow, corr) #B,input_dim,H,W
        inp = torch.cat([inp, motion_features], dim=1)  #B,2*input_dim,H,W
        net = self.gru(net, inp) #B,hidden_dim,H,W
        delta_flow = self.flow_head(net)

        return net, None, delta_flow

class BasicUpdateBlock(nn.Module):
    def __init__(self, hidden_dim=64, input_dim=64, corr_levels=3, corr_radius=4):
        super(BasicUpdateBlock, self).__init__()
        cor_planes = corr_levels * (2*corr_radius + 1)**2
        self.encoder = BasicMotionEncoder(cor_planes, hidden_dim-2)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=input_dim+hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)

        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9, 1, padding=0))

    def forward(self, net, inp, corr, flow, upsample=True):
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)

        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)

        # scale mask to balence gradients
        # mask = .25 * self.mask(net)
        return net, None, delta_flow

class SmallNetUpdateBlock(nn.Module):
    def __init__(self, hidden_dim=64, input_dim=64, cor_plane=243):
        super(SmallNetUpdateBlock, self).__init__()
        # self.encoder = SmallMotionEncoder(input_dim+hidden_dim, input_dim-2) #outdim = input_dim
        self.encoder = SmallMotionEncoder(cor_plane, hidden_dim-2) #outdim = hidden_dim
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=input_dim+hidden_dim)
        # self.gru = ConvGRU(hidden_dim=hidden_dim, input_dim=input_dim+hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=128)

    def forward(self, net, inp, corr, flow):
        motion_features = self.encoder(flow, corr) #B,input_dim,H,W
        inp = torch.cat([inp, motion_features], dim=1)  #B,2*input_dim,H,W
        net = self.gru(net, inp) #B,hidden_dim,H,W
        delta_flow = self.flow_head(net)

        return net, None, delta_flow
    
class NetUpdateBlock(nn.Module):
    def __init__(self, hidden_dim=64, input_dim=64, cor_plane=243, corr_levels=3, corr_radius=4):
        super(NetUpdateBlock, self).__init__()
        self.encoder = BasicMotionEncoder(cor_plane, hidden_dim-2)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=input_dim+hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)

    def forward(self, net, inp, corr, flow, upsample=True):
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)

        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)

        return net, None, delta_flow
    
# class FmapMaskBlock(nn.Module):
#     def __init__(self, hidden_dim=128, input_dim=192+128):
#         super(FmapMaskBlock, self).__init__()
#         self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
#         self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
#         self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)

#     def forward(self, fmaps, x):
#         for h in fmaps:
#             _,_,height,weight = h.shape
#             x = F.interpolate(x, size=(height, weight), mode='bilinear', align_corners=False)

#             hx = torch.cat([h, x], dim=1) #B C+2 H W

#             z = torch.sigmoid(self.convz(hx))
#             r = torch.sigmoid(self.convr(hx))
#             q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))

#             h = (1-z) * h + z * q
#         return fmaps


    
# class BasicUpdateBlock(nn.Module):
#     def __init__(self, args, hidden_dim=128, input_dim=128):
#         super(BasicUpdateBlock, self).__init__()
#         self.args = args
#         self.encoder = BasicMotionEncoder(args)
#         self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128+hidden_dim)
#         self.flow_head = FlowHead(hidden_dim, hidden_dim=256)

#         self.mask = nn.Sequential(
#             nn.Conv2d(128, 256, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 64*9, 1, padding=0))

#     def forward(self, net, inp, corr, flow, upsample=True):
#         motion_features = self.encoder(flow, corr)
#         inp = torch.cat([inp, motion_features], dim=1)

#         net = self.gru(net, inp)
#         delta_flow = self.flow_head(net)

#         # scale mask to balence gradients
#         mask = .25 * self.mask(net)
#         return net, mask, delta_flow

