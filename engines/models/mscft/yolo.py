# YOLOv5 YOLO-specific modules

import argparse
import logging
import sys
from copy import deepcopy
from pathlib import Path

sys.path.append(Path(__file__).parent.parent.absolute().__str__())  # to run '$ python *.py' files in subdirectories
logger = logging.getLogger(__name__)

from engines.models.mscft.common import *
from engines.models.mscft.experimental import *
from engines.models.mscft.utils.autoanchor import check_anchor_order
from engines.models.mscft.utils.general import make_divisible, check_file, set_logging
from engines.models.mscft.utils.torch_utils import time_synchronized, fuse_conv_and_bn, model_info, scale_img, initialize_weights, \
    select_device, copy_attr

try:
    import thop  # for FLOPS computation
except ImportError:
    thop = None


class Detect(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(Detect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class Model(nn.Module):
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        super(Model, self).__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict

        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.safe_load(f)  # model dict
            print("YAML")
            print(self.yaml)

        # Define model
        ch_in = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels (int or list for dual stream)
        if isinstance(ch_in, (list, tuple)):
            rgb = ch_in[0]
            ms = ch_in[1] if len(ch_in) > 1 else ch_in[0]
            # 模型内部使用至少 3 通道的 MS 分支以兼容 GPT/Focus
            ms_branch = max(3, ms)
            ch_list = [rgb, ms_branch]  # 双流输入顺序：RGB，MS
            self.rgb_ch = rgb
            self.ms_ch = ms
            self.ms_branch_ch = ms_branch
        else:
            ch_list = [ch_in]
            self.rgb_ch = ch_in
            self.ms_ch = ch_in
            self.ms_branch_ch = ch_in
        if nc and nc != self.yaml['nc']:
            logger.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            logger.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        # 多模态结构存在双流，初始化为双输入通道列表以兼容 -4 等引用
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch_list)  # model, savelist
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        # logger.info([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))])

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        # print("Detect")
        # print(m)
        if isinstance(m, Detect):
            s = 256  # 2x min stride
            if isinstance(ch_in, (list, tuple)):
                rgb = torch.zeros(1, self.rgb_ch, s, s)
                ms = torch.zeros(1, self.ms_ch, s, s)
                dummy = (rgb, ms)
            else:
                dummy = torch.zeros(1, ch_in, s, s)
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(dummy)])  # forward
            # print("m.stride", m.stride)
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()  # only run once
            # logger.info('Strides: %s' % m.stride.tolist())

        # Init weights, biases
        initialize_weights(self)
        self.info()
        logger.info('')

    def forward(self, x, augment=False, profile=False):
        # 若传入 (rgb, ir) 双流输入，分别记录两路输入
        ms_input = None
        if isinstance(x, (tuple, list)):
            x0 = x[0]
            ms_input = x[1] if len(x) > 1 else x[0]
            # 将 MS 通道扩展到模型期望的分支通道数
            if ms_input.shape[1] != getattr(self, "ms_branch_ch", ms_input.shape[1]):
                repeat = self.ms_branch_ch // ms_input.shape[1]
                if repeat > 1:
                    ms_input = ms_input.repeat(1, repeat, 1, 1)
                elif ms_input.shape[1] < self.ms_branch_ch:
                    pad = self.ms_branch_ch - ms_input.shape[1]
                    ms_input = torch.cat([ms_input, ms_input[:, :pad]], dim=1)
        else:
            x0 = x
        if augment:
            img_size = x.shape[-2:]  # height, width
            s = [1, 0.83, 0.67]  # scales
            f = [None, 3, None]  # flips (2-ud, 3-lr)
            y = []  # outputs
            for si, fi in zip(s, f):
                xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
                yi = self.forward_once(xi)[0]  # forward
                # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
                yi[..., :4] /= si  # de-scale
                if fi == 2:
                    yi[..., 1] = img_size[0] - yi[..., 1]  # de-flip ud
                elif fi == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0]  # de-flip lr
                y.append(yi)
            return torch.cat(y, 1), None  # augmented inference, train
        else:
            return self.forward_once(x0, profile, ms_input=ms_input)  # single-scale inference, train

    def forward_once(self, x, profile=False, y_init=None, ms_input=None):
        y = y_init or []
        dt = []  # outputs
        x0 = x  # 原始输入，用于超范围回退
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                if isinstance(m.f, int):
                    idx = m.f
                    if idx < -len(y) or idx >= len(y):
                        x = x0
                    else:
                        x = y[idx]
                else:
                    inputs = []
                    for j in m.f:
                        if j == -1:
                            inputs.append(x)
                        else:
                            idx = j
                            if idx < -len(y) or idx >= len(y):
                                inputs.append(x0)
                            else:
                                inputs.append(y[idx])
                    x = inputs  # from earlier layers

            if profile:
                o = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPS
                t = time_synchronized()
                for _ in range(10):
                    _ = m(x)
                dt.append((time_synchronized() - t) * 100)
                if m == self.model[0]:
                    logger.info(f"{'time (ms)':>10s} {'GFLOPS':>10s} {'params':>10s}  {'module'}")
                logger.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')

            # 特殊处理：双流分支的第二个 Focus 使用 ir/ms 输入
            if isinstance(m, Focus) and getattr(m, "f", None) == -4 and ms_input is not None:
                x = ms_input

            x = m(x)  # run
            # 始终保留各层输出，适配包含负索引的跨层引用
            y.append(x)

        if profile:
            logger.info('%.1fms total' % sum(dt))
        return x

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            logger.info(
                ('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             logger.info('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        logger.info('Fusing layers... ')
        for m in self.model.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.fuseforward  # update forward
        self.info()
        return self

    def nms(self, mode=True):  # add or remove NMS module
        present = type(self.model[-1]) is NMS  # last layer is NMS
        if mode and not present:
            logger.info('Adding NMS... ')
            m = NMS()  # module
            m.f = -1  # from
            m.i = self.model[-1].i + 1  # index
            self.model.add_module(name='%s' % m.i, module=m)  # add
            self.eval()
        elif not mode and present:
            logger.info('Removing NMS... ')
            self.model = self.model[:-1]  # remove
        return self

    def autoshape(self):  # add autoShape module
        logger.info('Adding autoShape... ')
        m = autoShape(self)  # wrap model
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())  # copy attributes
        return m

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)


def parse_model(d, ch):  # model_dict, input_channels list for dual stream
    logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    # ch 为原始输入通道列表（RGB, MS），不在构图过程中修改
    base_ch = list(ch)
    layers, save, out_ch = [], [], []

    def get_ch(idx: int) -> int:
        """解析 from 索引对应的通道数，支持 -4 映射到 MS 输入。"""
        # 特殊：-4 表示 MS 分支原始输入
        if idx == -4 and len(base_ch) > 1:
            return base_ch[1]
        # 第一层使用 RGB 输入
        if idx == -1 and len(out_ch) == 0:
            return base_ch[0]
        if idx < 0:
            real_idx = idx
            if abs(real_idx) <= len(out_ch):
                return out_ch[real_idx]
            # 兜底返回 RGB 输入通道
            return base_ch[0]
        if idx < len(out_ch):
            return out_ch[idx]
        # 超出范围时退回原始输入
        if idx < len(base_ch):
            return base_ch[idx]
        raise IndexError(f"parse_model channel index out of range: f={idx}, len(out_ch)={len(out_ch)}, base={base_ch}")

    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except Exception:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, DWConv, MixConv2d, Focus, CrossConv, BottleneckCSP,
                 C3, C3TR]:
            c1 = get_ch(f if isinstance(f, int) else f[0])
            c2 = args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in [BottleneckCSP, C3, C3TR]:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            c2 = get_ch(f)
            args = [c2]
        elif m is Add:
            c2 = get_ch(f[0])
            args = [c2]
        elif m is Add2:
            c2 = get_ch(f[0])
            args = [c2, args[1]]
        elif m is GPT:
            c2 = get_ch(f[0])
            args = [c2]
        elif m is Concat:
            c2 = sum(get_ch(x) for x in f)
        elif m is Detect:
            args.append([get_ch(x) for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
            # Detect 的输出通道仅用于日志，占位设置为 no
            c2 = no
        elif m is Contract:
            c2 = get_ch(f) * args[0] ** 2
        elif m is Expand:
            c2 = get_ch(f) // args[0] ** 2
        else:
            c2 = get_ch(f)

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        out_ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    set_logging()
    device = select_device(opt.device)

    # Create model
    model = Model(opt.cfg).to(device)
    input_rgb = torch.Tensor(8, 3, 640, 640).to(device)
    output = model(input_rgb)

    # print(model)
    # model.train()
    # torch.save(model, "yolov5s.pth")

    # Profile
    # img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 320, 320).to(device)
    # y = model(img, profile=True)

    # Tensorboard (not working https://github.com/ultralytics/yolov5/issues/2898)
    # from torch.utils.tensorboard import SummaryWriter
    # tb_writer = SummaryWriter('.')
    # logger.info("Run 'tensorboard --logdir=models' to view tensorboard at http://localhost:6006/")
    # tb_writer.add_graph(torch.jit.trace(model, img, strict=False), [])  # add model graph
    # tb_writer.add_image('test', img[0], dataformats='CWH')  # add model to tensorboard
