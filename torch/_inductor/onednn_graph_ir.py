import dataclasses
import torch
from torch._inductor.ir import ExternKernelAlloc, FixedLayout, MultiOutput, MultiOutputLayout, ExternKernelNode
from typing import (
    Any,
    Sequence,
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

@dataclasses.dataclass
class OneDNNGraphFallbackKernel(ExternKernelAlloc):
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
        self.use_cpp_op_schema = False

        self.op_overload = kernel

        assert isinstance(
            kernel,
            (
                torch._ops.OpOverload,
                torch._ops.HigherOrderOperator,
                torch._inductor.fx_passes.onednn_graph_fusion.OnednnGraphPartitionModule
            ),
        ), f"Fails to create OneDNNGraphFallbackKernel for {kernel}: {type(kernel)} not supported"

        if kernel.__module__ == "torch._ops.aten":
            op_base_name = (
                kernel.__name__.split(".")[0]
                if isinstance(kernel, torch._ops.OpOverload)
                else kernel.__name__
            )
            if V.graph.cpp_wrapper:
                assert isinstance(kernel, torch._ops.OpOverload)
                # Calling with the default kernel name can lead to ambiguous behavior like the following example.
                # repeat_interleave(const at::Tensor & repeats, c10::optional<int64_t> output_size=c10::nullopt)
                # repeat_interleave(const at::Tensor & self, int64_t repeats,
                #       c10::optional<int64_t> dim=c10::nullopt, c10::optional<int64_t> output_size=c10::nullopt)
                self.kernel = (
                    f"at::{op_base_name}"
                    if kernel._overloadname == "default"
                    else f"at::_ops::{kernel.__name__.replace('.', '_')}::call"
                )
                schema = kernel._schema
            else:
                self.kernel = f"aten.{op_base_name}"

            if schema is not None:
                self.args_default_value = [
                    {"type": x.real_type, "value": x.default_value}
                    for x in schema.arguments
                    if not x.kwarg_only
                ]
                self.ordered_kwargs_for_cpp_kernel = [
                    x.name for x in schema.arguments if x.kwarg_only
                ]
                self.kwargs_default_value = {
                    x.name: {"type": x.real_type, "value": x.default_value}
                    for x in schema.arguments
                    if x.kwarg_only
                }
        elif isinstance(kernel, torch._ops.HigherOrderOperator):
            if getattr(torch._prims.rng_prims, kernel.__name__, None) is kernel:
                self.kernel = f"torch._prims.rng_prims.{kernel.__name__}"
            else:
                raise NotImplementedError(
                    "Unable to find HigherOrderOperator kernel name"
                )
        elif getattr(kernel, "is_opaque", False):
            self.kernel = kernel
        else:
            if V.graph.cpp_wrapper:
                self.use_cpp_op_schema = True
                self.set_cpp_kernel(kernel)
            else:
                self.kernel = (
                    f"{kernel.__module__.replace('._ops.', '.ops.')}.{kernel.__name__}"
                )
        self.unflatten_args = unflatten_args
        self.kwargs = {} if kwargs is None else kwargs
        V.graph.warn_fallback(self.kernel)

    def set_cpp_kernel(self, kernel):
        from torch._inductor.codegen.wrapper import get_cpp_op_schema

        assert (
            not kernel._schema.is_mutable
        ), f"mutable {kernel.__name__} is not supported with cpp_wrapper"

        # These checks are here because ops that return aliasing tensors will
        # return type Tensor& instead of Tensor, but codegen will always write
        # type Tensor on the LHS.
        def is_not_write(arg):
            return arg.alias_info is None or not arg.alias_info.is_write

        assert all(
            is_not_write(x) for x in kernel._schema.arguments
        ), f"{kernel.__name__} with alias_info arguments is not supported with cpp_wrapper"
        assert all(
            is_not_write(x) for x in kernel._schema.returns
        ), f"{kernel.__name__} with alias_info returns is not supported with cpp_wrapper"

        self.kernel = kernel._schema.name
        self.cpp_kernel_overlad_name = kernel._schema.overload_name
        self.cpp_kernel_key = (
            f"{self.kernel.replace('::', '_')}_{self.cpp_kernel_overlad_name}"
        )

        self.cpp_op_schema = get_cpp_op_schema(kernel)
        self.ordered_kwargs_for_cpp_kernel = [
            x.name for x in kernel._schema.arguments if x.kwarg_only
        ]

    def get_arg_default_value(self, pos):
        assert hasattr(
            self, "args_default_value"
        ), "self.args_default_value has to be provided"
        assert pos < len(
            self.args_default_value
        ), f"expected the index {pos} to be smaller than len(self.args_default_value): {len(self.args_default_value)}"
        return self.args_default_value[pos]["value"]

    def codegen_args(self):
        @dataclasses.dataclass
        class Shim:
            ref: Any

            def __repr__(self):
                return self.ref

        tensor_args = [Shim(x.codegen_reference()) for x in self.inputs]
        args, kwargs = self.unflatten_args(tensor_args, self.constant_args)
        args = [V.graph.wrapper_code.val_to_arg_str(x) for x in args]
        # Previously, we want to maintain forward-compatibility by skipping
        # default args in the serialized artifacts in fbcode. However,
        # some of our shim interfaces require default values being set.
        # Discussed with Sherlock offline and we decided to allow serializing
        # default args into the C++ wrapper code for now. We will refine this
        # part if we see real FC requirement. More details related to FC
        # can be found at:
        # https://docs.google.com/document/d/1FzWm-sHYwmRi3x_g036kOxd99KaYquUsA-L5JwOn8ys/edit?usp=sharing
        if V.graph.cpp_wrapper and hasattr(self, "args_default_value"):
            n_args = len(args)
            n_pos_args = len(self.args_default_value)
            # Some positional args are not provided, need to use their default value in cpp wrapper
            if n_args < n_pos_args:
                pos_args = [
                    self.get_arg_default_value(i) for i in range(n_args, n_pos_args)
                ]
                pos_args = [V.graph.wrapper_code.val_to_arg_str(x) for x in pos_args]
                args.extend(pos_args)

        # let self.codegen_kwargs handle kwargs
        self.kwargs.update(kwargs)
        return args

    @staticmethod
    def find_device(tensor_args, example_output):
        if tensor_args:
            return tensor_args[0].get_device()
        if isinstance(example_output, torch.Tensor):
            return example_output.device
        if isinstance(example_output, (list, tuple)):
            devices = {OneDNNGraphFallbackKernel.find_device(None, x) for x in example_output}
            # Remove None
            devices = [device for device in devices if device]
            if len(devices) == 1:
                return devices[0]
            for device in devices:
                if device.type == "cuda":
                    return device
            return devices[0]
        return None

    def has_side_effects(self):
        # TODO - some fallbacks are still OpOverloadPackets
        if not isinstance(self.op_overload, torch._ops.OpOverload):
            return False
        return get_schema_info(self.op_overload).is_mutable()

    def has_aliasing(self):
        # TODO - some fallbacks are still OpOverloadPackets
        if not isinstance(self.op_overload, torch._ops.OpOverload):
            return False
        return torch._inductor.utils.is_view(self.op_overload)

    # ProxyExecutor Design Note
    # We export the ExternFallbackNodes (for custom ops) into a serialized file
    # and run it with a host side proxy executor to address the ABI problem
    # This is currently only implemented for fbcode. Eventually, we will also make this work for OSS.
    # Detailed design doc can be found at
    # https://docs.google.com/document/d/1wC4DOZFaYym2t1Esz0X5yxlLI3RDnSiyRbUus3bkJ64/edit?usp=sharing
    def export_extern_kernel_node(self):
        assert isinstance(self, OneDNNGraphFallbackKernel)
        args, kwargs = self.unflatten_args(self.inputs, self.constant_args)
        ordered_kwargs = [
            kwargs.get(key, None) for key in self.ordered_kwargs_for_cpp_kernel
        ]

        serializer = GraphModuleSerializer(None, None)
        named_arguments = serializer.serialize_inputs(self.op_overload, args, kwargs)

        # serialize_outputs
        def handle_single_output(return_type, output):
            if isinstance(return_type, torch.TensorType):
                # For single Tensor
                out = output
                if isinstance(output, (list, tuple)):
                    assert len(output) == 1
                    out = output[0]
                return export_schema.Argument.create(
                    as_tensor=export_schema.TensorArgument(name=out.get_name())
                )
            elif isinstance(return_type, torch.ListType) and isinstance(
                return_type.getElementType(), torch.TensorType
            ):
                # For single TensorList
                return export_schema.Argument.create(
                    as_tensors=[
                        export_schema.TensorArgument(name=out.get_name())
                        for out in output
                    ]
                )
            else:
                raise RuntimeError("Unsupported return type")

        target = self.op_overload
        returns = target._schema.returns
        if len(returns) == 1:
            return_type = returns[0].real_type
            output_arguments = [handle_single_output(return_type, self.outputs)]
        else:
            # For tuple returns, e.g "-> (Tensor, Tensor)" or "-> (Tesnor, Tensor[])"
            assert isinstance(self.outputs, tuple)
            assert len(returns) == len(self.outputs)
            output_arguments = [
                handle_single_output(return_schema.real_type, output)
                for return_schema, output in zip(returns, self.outputs)
            ]

        node = ExternKernelNode(
            name=self.get_name(),
            node=export_schema.Node(
                target=self.kernel,
                inputs=named_arguments,
                outputs=output_arguments,
                metadata={},
            ),
        )

        V.graph.extern_kernel_nodes.append(node)

        return [*args, *ordered_kwargs]

    def codegen(self, wrapper):
        if self.use_cpp_op_schema:
            self.codegen_comment(wrapper)

            exported_args = None
            args = None
            if config.is_fbcode() and V.graph.cpp_wrapper:
                exported_args = self.export_extern_kernel_node()
            else:
                args = [*self.codegen_args(), *self.codegen_kwargs()]

            wrapper.generate_extern_kernel_alloc_and_find_schema_if_needed(
                self.get_name(),
                self.kernel,
                args,
                self.cpp_op_schema,
                self.cpp_kernel_key,
                self.cpp_kernel_overlad_name,
                self.op_overload,
                exported_args,
                self.outputs,
            )
        else:
            super().codegen(wrapper)

    @classmethod
    def create(cls, kernel, *args, **kwargs):
        fake_incorrect_kernels = (aten._fused_moving_avg_obs_fq_helper_functional,)
        context = (
            V.graph.fake_mode if kernel not in fake_incorrect_kernels else nullcontext()
        )
        with context:
            (
                example_output,
                tensor_args,
                non_tensor_args,
                unflatten_args,
                schema,
            ) = cls.process_kernel(kernel, *args, **kwargs)

        device = OneDNNGraphFallbackKernel.find_device(tensor_args, example_output)
        assert device, "Not sure where to find device info"

        def tensor_to_layout(output: torch.Tensor):
            return FixedLayout(
                output.device,
                output.dtype,
                convert_shape_to_inductor(output.size()),
                convert_shape_to_inductor(output.stride()),
            )

        packed = OneDNNGraphFallbackKernel(
            MultiOutputLayout(device),
            kernel,
            tensor_args,
            non_tensor_args,
            unflatten_args,
            schema=schema,
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
                    tensor_to_layout(output),
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
                ), f"OneDNNGraphFallbackKernel output type {type(output)} is not supported"
                return None

        outputs = generate_output(example_output, [])
        if isinstance(outputs, (list, tuple, dict)):
            packed.outputs = outputs  # type: ignore[assignment]
        else:
            packed.outputs = [outputs]
        return outputs

    def apply_constraint(self):
        return super().apply_constraint()
