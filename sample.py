import torch
import resnet
import copy
import helper
import torch.nn as nn

class QuantizedResNet18(nn.Module):
    def __init__(self, model_fp32):
        super(QuantizedResNet18, self).__init__()
        # QuantStub converts tensors from floating point to quantized.
        # This will only be used for inputs.
        self.quant = torch.quantization.QuantStub()
        # DeQuantStub converts tensors from quantized to floating point.
        # This will only be used for outputs.
        self.dequant = torch.quantization.DeQuantStub()
        # FP32 model
        self.model_fp32 = model_fp32

    def forward(self, x):
        # manually specify where tensors will be converted from floating
        # point to quantized in the quantized model
        x = self.quant(x)
        x = self.model_fp32(x)
        # manually specify where tensors will be converted from quantized
        # to floating point in the quantized model
        x = self.dequant(x)
        return x


# create a model instance
model_fp32 = resnet.resnet50(pretrained=True)


fused_model = copy.deepcopy(model_fp32)

# model must be set to eval mode for static quantization logic to work
fused_model.eval()
print(fused_model)

# attach a global qconfig, which contains information about what kind
# of observers to attach. Use 'fbgemm' for server inference and
# 'qnnpack' for mobile inference. Other quantization configurations such
# as selecting symmetric or assymetric quantization and MinMax or L2Norm
# calibration techniques can be specified here.

# Fuse the activations to preceding layers, where applicable.
# This needs to be done manually depending on the model architecture.
# Common fusions include `conv + relu` and `conv + batchnorm + relu`
fused_model = torch.quantization.fuse_modules(fused_model, [["conv1", "bn1", "relu"]], inplace=True)
for module_name, module in fused_model.named_children():
    if "layer" in module_name:
        for basic_block_name, basic_block in module.named_children():
            torch.quantization.fuse_modules(basic_block, [["conv1", "bn1", "relu"], ["conv2", "bn2"]], inplace=True)
            for sub_block_name, sub_block in basic_block.named_children():
                if sub_block_name == "downsample":
                    torch.quantization.fuse_modules(sub_block, [["0", "1"]], inplace=True)

    
quantized_model = QuantizedResNet18(model_fp32=fused_model)

quantized_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')


# Prepare the model for static quantization. This inserts observers in
# the model that will observe activation tensors during calibration.
# model_fp32_prepared = torch.quantization.prepare(model_fp32_fused)
torch.quantization.prepare(quantized_model, inplace=True)

print('==== Done.')

# calibrate the prepared model to determine quantization parameters for activations
# in a real world setting, the calibration would be done with a representative dataset

# Convert the observed model to a quantized model. This does several things:
# quantizes the weights, computes and stores the scale and bias value to be
# used with each activation tensor, and replaces key operators with quantized
# implementations.
# model_int8 = torch.quantization.convert(model_fp32_prepared)
quantized_model = torch.quantization.convert(quantized_model, inplace=True)
quantized_model.eval()
print(quantized_model)



cpu_device = torch.device("cpu:0")
int8_cpu_inference_latency = helper.measure_inference_latency(model=quantized_model, device=cpu_device, input_size=(1,3,224,224), num_samples=100)
print(f"int8_cpu_inference_latency : {int8_cpu_inference_latency}")

fp32_cpu_inference_latency = helper.measure_inference_latency(model=model_fp32, device=cpu_device, input_size=(1,3,224,224), num_samples=100)
print(f"fp32_cpu_inference_latency : {fp32_cpu_inference_latency}")

exit()


# run the model, relevant calculations will happen in int8
input_fp32 = torch.randn(1, 3, 224, 224)
res = quantized_model(input_fp32)
print(res)


