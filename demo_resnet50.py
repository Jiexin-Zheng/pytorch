import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torch._inductor import config
from torch._inductor.fx_passes.ipex_onednn_graph_fusion import onednn_graph_fuse_fx
config.post_grad_custom_pre_pass = onednn_graph_fuse_fx
from torch._inductor.codegen.onednn_graph_wrapper import OneDNNGraphWrapperCodeGen
from torch._inductor.codegen.common import register_backend_for_device
from torch._inductor.codegen.cpp import CppScheduling
register_backend_for_device("cpu", CppScheduling, OneDNNGraphWrapperCodeGen)

# Load the pre-trained ResNet50 model
resnet50 = models.resnet50(pretrained=True)
resnet50.eval()  # Set the model to evaluation mode

# Create a dummy input tensor (batch size 1, 3 channels, height 224, width 224)
dummy_input = torch.randn(1, 3, 224, 224)

# Perform any necessary pre-processing on the dummy input
# For ResNet50, usually, you need to normalize the input using the same mean and std
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
preprocess = transforms.Compose([transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 normalize])
dummy_input = preprocess(Image.fromarray((dummy_input.squeeze().numpy().transpose((1, 2, 0)) * 255).astype('uint8')))

# Add an extra batch dimension to the input
dummy_input = dummy_input.unsqueeze(0)

resnet50_optimized = torch.compile(resnet50)

# Run the model on the dummy input
with torch.no_grad():
    output = resnet50_optimized(dummy_input)
    #output = resnet50(dummy_input)
# Print the model's prediction for the dummy input
print("Model Output:", output.argmax(dim=1))