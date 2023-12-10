import torch
from torch._inductor import config
from torch._inductor.fx_passes.ipex_onednn_graph_fusion import onednn_graph_fuse_fx
config.post_grad_custom_pre_pass = onednn_graph_fuse_fx
from torch._inductor.codegen.onednn_graph_wrapper import OneDNNGraphWrapperCodeGen
from torch._inductor.codegen.common import register_backend_for_device
from torch._inductor.codegen.cpp import CppScheduling
register_backend_for_device("cpu", CppScheduling, OneDNNGraphWrapperCodeGen)

class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.main = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=7, out_channels=3, kernel_size=(3, 3), padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=3, out_channels=2, kernel_size=(3, 3), padding=1),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(20000, 1)
        )

    def forward(self, x):
        out = self.main(x)
        return out

with torch.no_grad():
    mod = CNN()
    #opt_mod = torch.compile(mod, backend="onednn")
    opt_mod = torch.compile(mod, backend="inductor")
    input = torch.randn([1, 7, 100, 100],  dtype=torch.float32)
    print("opt module output:")
    print(opt_mod(input))