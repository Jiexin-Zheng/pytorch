import dataclasses
import torch
from torch._inductor.ir import FallbackKernel, ExternKernelAlloc, FixedLayout, MultiOutput, MultiOutputLayout, Layout
from typing import (
    Any,
    Sequence,
    List,
    Dict,
)
from torch._inductor.virtualized import ops, V
from torch._subclasses.fake_tensor import get_schema_info
from torch._export.serde.serialize import GraphModuleSerializer
from contextlib import nullcontext
import torch._export.serde.schema as export_schema
import torch._logging
from torch._export.serde.serialize import GraphModuleSerializer
from torch._subclasses.fake_tensor import get_schema_info
from torch._inductor import config
from torch._inductor.utils import (
    convert_shape_to_inductor,
)
aten = torch.ops.aten

class OneDNNGraphKernel(ExternKernelAlloc):
    def __init__(
        self,
        layout,
        kernel,
        tensor_args,
        nontensor_args,
        unflatten_args,
        kwargs=None,
        schema=None,
    ):
        super().__init__(
            layout,
            tuple(tensor_args),
            tuple(nontensor_args),
        )
        # We need output buffers for generating kernel arguments in the
        # abi-compatible mode, where we retrieve outputs by pass each individual
        # output through the abi-compatible interface.
        self.outputs: Sequence[Any] = []
        self.use_runtime_dispatch = False
        self.abi_compatible_kernel = None

        assert isinstance(
            kernel,
            torch._inductor.fx_passes.ipex_onednn_graph_fusion.OnednnGraphPartitionModule
        ), f"Fails to create OneDNNGraphKernel for {kernel}: {type(kernel)} not supported"
        self.kernel = kernel

        self.unflatten_args = unflatten_args
        self.kwargs = {} if kwargs is None else kwargs
        V.graph.warn_fallback(self.kernel)
    
    def codegen(self, wrapper):
        # kernel = self.op_overload
        # if isinstance(kernel, torch._inductor.fx_passes.ipex_onednn_graph_fusion.OnednnGraphPartitionModule):
        #     self.kernel = kernel
        # elif kernel.namespace == "aten":
        #     # Aten Fallback Ops
        #     assert isinstance(kernel, torch._ops.OpOverload)
        #     op_base_name = kernel.__name__.split(".")[0]

        #     if V.graph.cpp_wrapper:
        #         if config.is_fbcode() and kernel not in has_c_shim:
        #             log.warning(
        #                 "%s is missing a c-shim implementation, using proxy executor as fallback",
        #                 kernel,
        #             )
        #             self.use_runtime_dispatch = True
        #             self.set_cpp_kernel(kernel)
        #         else:
        #             # Calling with the default kernel name can lead to ambiguous behavior like the following example.
        #             # repeat_interleave(const at::Tensor & repeats, c10::optional<int64_t> output_size=c10::nullopt)
        #             # repeat_interleave(const at::Tensor & self, int64_t repeats,
        #             #       c10::optional<int64_t> dim=c10::nullopt, c10::optional<int64_t> output_size=c10::nullopt)
        #             self.cpp_kernel = (
        #                 f"at::{op_base_name}"
        #                 if kernel._overloadname == "default"
        #                 else f"at::_ops::{kernel.__name__.replace('.', '_')}::call"
        #             )
        #             schema = kernel._schema

        #             self.args_default_value = [
        #                 {"type": x.real_type, "value": x.default_value}
        #                 for x in schema.arguments
        #                 if not x.kwarg_only
        #             ]
        #             self.ordered_kwargs_for_cpp_kernel = [
        #                 x.name for x in schema.arguments if x.kwarg_only
        #             ]
        #             self.kwargs_default_value = {
        #                 x.name: {"type": x.real_type, "value": x.default_value}
        #                 for x in schema.arguments
        #                 if x.kwarg_only
        #             }
        #     else:
        #         self.kernel = f"aten.{op_base_name}"

        # elif isinstance(kernel, torch._ops.HigherOrderOperator):
        #     if getattr(torch._prims.rng_prims, kernel.__name__, None) is kernel:
        #         self.kernel = f"torch._prims.rng_prims.{kernel.__name__}"
        #     else:
        #         raise NotImplementedError(
        #             "Unable to find HigherOrderOperator kernel name"
        #         )
        # else:
        #     # For non-aten OpOverload, i.e. custom ops
        #     if V.graph.cpp_wrapper:
        #         self.use_runtime_dispatch = True
        #         self.set_cpp_kernel(kernel)
        #     else:
        #         self.kernel = (
        #             f"{kernel.__module__.replace('._ops.', '.ops.')}.{kernel.__name__}"
        #         )

        # if self.use_runtime_dispatch:
        #     self.codegen_comment(wrapper)

        #     exported_args = None
        #     args = None
        #     if config.is_fbcode() and V.graph.cpp_wrapper:
        #         exported_args = self.export_extern_kernel_node()
        #     else:
        #         args = [*self.codegen_args(), *self.codegen_kwargs()]

        #     wrapper.generate_extern_kernel_alloc_and_find_schema_if_needed(
        #         self.get_name(),
        #         self.codegen_kernel_name(),
        #         args,
        #         self.cpp_op_schema,
        #         self.cpp_kernel_key,
        #         self.cpp_kernel_overload_name,
        #         self.op_overload,
        #         exported_args,
        #         self.outputs,
        #     )
        # else:
        #     self.codegen_comment(wrapper)
        #     args = [*self.codegen_args(), *self.codegen_kwargs()]
        #     if isinstance(self.kernel, torch._inductor.fx_passes.ipex_onednn_graph_fusion.OnednnGraphPartitionModule):
        #         V.graph.wrapper_code.generate_opaque_kernel_alloc(self.get_name(), self.kernel.name(), args)
        #     else:
        #         V.graph.wrapper_code.generate_fallback_kernel(self, args)
        #     V.graph.wrapper_code.generate_opaque_kernel_alloc(self.get_name(), self.kernel.name(), args)
        #     if isinstance(self.layout, Layout):
        #         self.codegen_size_asserts(wrapper)

        self.codegen_comment(wrapper)
        args = [*self.codegen_args(), *self.codegen_kwargs()]
        V.graph.wrapper_code.generate_opaque_kernel_alloc(self.get_name(), self.kernel.name(), args)
        if isinstance(self.layout, Layout):
            self.codegen_size_asserts(wrapper)

    @classmethod
    def create(cls, kernel, *args, **kwargs):
        # fake_incorrect_kernels = (aten._fused_moving_avg_obs_fq_helper_functional,)
        # context = (
        #     V.graph.fake_mode if kernel not in fake_incorrect_kernels else nullcontext()
        # )
        context = V.graph.fake_mode
        with context:
            (
                example_output,
                tensor_args,
                non_tensor_args,
                unflatten_args,
                schema,
            ) = cls.process_kernel(kernel, *args, **kwargs)

        device = cls.find_device(tensor_args, example_output)
        assert device, "Not sure where to find device info"

        packed = cls(
            MultiOutputLayout(device),
            kernel,
            tensor_args,
            non_tensor_args,
            unflatten_args,
        )

        def generate_output(output, indices):
            if isinstance(output, (list, tuple)):
                return type(output)(
                    generate_output(output[i], indices + [(type(output), i)])
                    for i in range(len(output))
                )
            elif isinstance(output, dict):
                return {
                    key: generate_output(val, indices + [(type(output), key)])
                    for key, val in output.items()
                }
            elif isinstance(output, torch.Tensor):
                return MultiOutput(
                    cls.tensor_to_layout(output),
                    packed,
                    indices,
                )
            elif isinstance(output, int):
                return output
            elif isinstance(output, torch.SymInt):
                return output.node.expr
            else:
                assert (
                    output is None
                ), f"FallbackKernel output type {type(output)} is not supported"
                return None

        outputs = generate_output(example_output, [])
        if isinstance(outputs, (list, tuple, dict)):
            packed.outputs = outputs  # type: ignore[assignment]
        else:
            packed.outputs = [outputs]
        return outputs

    @staticmethod
    def find_device(tensor_args, example_output):
        if tensor_args:
            return tensor_args[0].get_device()
        if isinstance(example_output, torch.Tensor):
            return example_output.device
        if isinstance(example_output, (list, tuple)):
            devices = {OneDNNGraphKernel.find_device(None, x) for x in example_output}
            # Remove None
            devices = [device for device in devices if device]
            if len(devices) == 1:
                return devices[0]
            for device in devices:
                if device.type == "cuda":
                    return device
            return devices[0]
        return None

    @staticmethod
    def tensor_to_layout(output: torch.Tensor):
        return FixedLayout(
            output.device,
            output.dtype,
            convert_shape_to_inductor(output.size()),
            convert_shape_to_inductor(output.stride()),
        )