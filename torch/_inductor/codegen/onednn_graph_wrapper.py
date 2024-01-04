from torch._inductor.codegen.wrapper import WrapperCodeGen
from .. import codecache 

class OneDNNGraphWrapperCodeGen(WrapperCodeGen):
    def __init__(self):
        super().__init__()

    def write_header(self):
        self.header.splice(
            f"""
                from ctypes import c_void_p, c_long
                import torch
                import math
                import random
                import os
                import tempfile
                from math import inf, nan
                from torch._inductor.hooks import run_intermediate_hooks
                from torch._inductor.utils import maybe_profile

                from torch import device, empty, empty_strided
                from {codecache.__name__} import AsyncCompile
                from torch._inductor.select_algorithm import extern_kernels
                from torch._inductor.fx_passes.ipex_onednn_graph_fusion import global_opaque_ops
                global_dict = globals()
                for name, value in global_opaque_ops.items():
                   global_dict[name] = value
                aten = torch.ops.aten
                assert_size_stride = torch._C._dynamo.guards.assert_size_stride
                reinterpret_tensor = torch.ops.inductor._reinterpret_tensor
                async_compile = AsyncCompile()

            """
        )

    # Since opaque kernels may not have origin node available, this method is
    # the opaque node equivalent for generate_extern_kernel_alloc()
    def generate_opaque_kernel_alloc(self, extern_kernel_name, opaque_kernel, args):
        self.writeline(
            f"{self.declare}{extern_kernel_name} = {opaque_kernel}({', '.join(args)}){self.ending}"
        )
