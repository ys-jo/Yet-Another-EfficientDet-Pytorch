import torch
import os
from backbone import EfficientDetBackbone
import yaml


class Params:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)

if __name__ == "__main__":
    params = Params(f'projects/coco.yml')
    model = EfficientDetBackbone(num_classes=len(params.obj_list), compound_coef=0,
                                 ratios=eval(params.anchors_ratios), scales=eval(params.anchors_scales))
    model = model.cuda()
    ret = model.load_state_dict(torch.load("./test.pth"), strict=False)
    dummy_input = torch.randn(1, 3, 512, 512, device='cuda')
     
    model.set_swish(memory_efficient=False)
    torch.onnx.export(model, dummy_input, "test.onnx", opset_version=11)
    print("export onnx file")
