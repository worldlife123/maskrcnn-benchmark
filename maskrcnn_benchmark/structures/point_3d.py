import torch
import math

# transpose
FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1

class PointDepth(object):
    def __init__(self, depths, size, focal_length=None, baseline=None, min_value=None, max_value=None, mode="disp"):
        # modes: "disp", "disp_base", "disp_unity", "disp_base_unity", "disp_256", "depth", "inv_depth"

        device = depths.device if isinstance(depths, torch.Tensor) else torch.device('cpu')
        depths = torch.as_tensor(depths, dtype=torch.float32, device=device)
        # depths = depths.view(-1, 1)
        
        self.depths = depths# [..., :2]

        self.size = size
        self.focal_length = focal_length
        self.baseline = baseline
        self.max_value = max_value
        self.min_value = min_value
        
        if mode not in ["disp", "disp_base", "disp_unity", "disp_base_unity", "disp_256", "depth", "inv_depth", "depth_linear"]:
            raise ValueError('mode should be in ["disp", "disp_base", "disp_unity", "disp_base_unity", "disp_256", "depth", "inv_depth", "depth_linear"]')

        if mode in ["depth", "inv_depth", "depth_linear"] and focal_length is None:
            raise ValueError('depth modes should have focal_length')

        if mode in ["depth", "inv_depth", "disp_base", "disp_base_unity", "depth_linear"] and baseline is None:
            raise ValueError('depth and disp_base modes should have baseline')

        if mode in ["depth_linear"] and (min_value is None or max_value is None):
            raise ValueError('depth_linear modes should have min_value and max_value')

        self.mode = mode

        # interpolate
        # if not interpolate_mode in ["none", "linear"]:
        #     raise ValueError('interpolate_mode should be in ["none", "linear"]')
        # self.interpolate_mode = "none" # interpolate_mode
        # self.interpolate_params = {} # interpolate_params
        # self.min_value = interpolate_range[0]
        # params["max_value"] = interpolate_range[1]
        # if not interpolate_mode in ["none"]:
        #     self._interpolate(interpolate_mode, interpolate_params)

        self.extra_fields = {}

    def _interpolate(self, mode, params):
        if not mode in ["none", "linear"]:
            raise ValueError('interpolate_mode should be in ["none", "linear"]')
        if mode in ['linear'] and not (params.get("min_value") and params.get("max_value")):
            raise ValueError('linear interpolation requires min, max')
        if mode == self.interpolate_mode:
            return

        # first convert to "none" mode
        if self.interpolate_mode == "linear":
            depths = params["min_value"] + self.depths * (params["max_value"] - params["min_value"])

        # then convert to required mode
        if mode == "linear":
            self.depths = (depths - params["min_value"]) / (params["max_value"] - params["min_value"])

        # finally set flags
        self.interpolate_mode = mode
        self.interpolate_params = params

    
    def convert(self, mode):
        if mode not in ["disp", "disp_base", "disp_unity", "disp_base_unity", "disp_256", "depth", "inv_depth", "depth_linear"]:
            raise ValueError('mode should be in ["disp", "disp_base", "disp_unity", "disp_base_unity", "disp_256", "depth", "inv_depth", "depth_linear"]')
        if mode == self.mode:
            return self

        # convert to disp
        if self.mode == "disp":
            disps = self.depths
        elif self.mode == "disp_base":
            disps = self.depths * self.baseline
        elif self.mode == "disp_unity":
            disps = self.depths * self.size[0]
        elif self.mode == "disp_base_unity":
            disps = self.depths * self.size[0] * self.baseline
        elif self.mode == "disp_256":
            disps = self.depths * 256
        elif self.mode == "depth":
            disps = self.baseline * self.focal_length / self.depths
        elif self.mode == "inv_depth":
            disps = self.depths * self.focal_length * self.baseline
        elif self.mode == "depth_linear":
            disps = self.baseline * self.focal_length / (self.min_value + self.depths * (self.max_value - self.min_value))

        if mode == "disp":
            new_depths = disps
        elif mode == "disp_base":
            new_depths = disps / self.baseline
        elif mode == "disp_unity":
            new_depths = disps / self.size[0]
        elif mode == "disp_base_unity":
            new_depths = disps / self.size[0] / self.baseline
        elif mode == "disp_256":
            new_depths = disps / 256
        elif mode == "depth":
            new_depths = self.baseline * self.focal_length / disps 
        elif mode == "inv_depth":
            new_depths = disps / self.focal_length / self.baseline
        elif mode == "depth_linear":
            new_depths = (self.baseline * self.focal_length / disps - self.min_value) / (self.max_value - self.min_value)
        new_depths = type(self)(new_depths, self.size, focal_length=self.focal_length, baseline=self.baseline, min_value=self.min_value, max_value=self.max_value, mode=mode)
        for k, v in self.extra_fields.items():
            new_depths.add_field(k, v)

        return new_depths

    def crop(self, box):
        raise NotImplementedError()

    def resize(self, size, *args, **kwargs):
        s = size[0] / self.size[0] # set x scale
        # convert to disp mode then scale
        new_depths = self.convert("disp").depths * s
        new_depths = type(self)(new_depths, size, focal_length=(self.focal_length*s), baseline=self.baseline, min_value=self.min_value, max_value=self.max_value, mode="disp")
        for k, v in self.extra_fields.items():
            new_depths.add_field(k, v)
        return new_depths.convert(self.mode)

    def transpose(self, method):
        # nothing to change
        return self

    def affine(self, t, s):
        new_depths = self.convert("disp").depths * s
        new_depths = type(self)(new_depths, self.size, focal_length=self.focal_length, baseline=self.baseline, min_value=self.min_value, max_value=self.max_value, mode="disp")
        for k, v in self.extra_fields.items():
            new_depths.add_field(k, v)
        return new_depths.convert(self.mode)

    def to(self, *args, **kwargs):
        depths = type(self)(self.depths.to(*args, **kwargs), self.size, focal_length=self.focal_length, baseline=self.baseline, min_value=self.min_value, max_value=self.max_value, mode=self.mode)
        for k, v in self.extra_fields.items():
            if hasattr(v, "to"):
                v = v.to(*args, **kwargs)
            depths.add_field(k, v)
        return depths

    def __getitem__(self, item):
        depths = type(self)(self.depths[item], self.size, focal_length=self.focal_length, baseline=self.baseline, min_value=self.min_value, max_value=self.max_value, mode=self.mode)
        for k, v in self.extra_fields.items():
            depths.add_field(k, v[item])
        return depths # self.depths[item]

    def __len__(self):
        return self.depths.shape[0]

    def add_field(self, field, field_data):
        self.extra_fields[field] = field_data

    def get_field(self, field):
        return self.extra_fields[field]

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += 'num_instances={}, '.format(len(self.depths))
        s += 'image_width={}, '.format(self.size[0])
        s += 'image_height={})'.format(self.size[1])
        s += "extra_fields={})".format(self.extra_fields.keys())
        return s