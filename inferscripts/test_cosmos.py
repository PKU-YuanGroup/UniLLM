from tqdm import tqdm
import numpy as np
import torch
from janus.models.cosmos_tokenizer.video_lib import CausalVideoTokenizer

_UINT8_MAX_F = float(torch.iinfo(torch.uint8).max)
_DTYPE, _DEVICE = torch.bfloat16, "cuda"

def tensor2numpy(input_tensor: torch.Tensor, range_min: int = -1) -> np.ndarray:
    
    """Converts tensor in [-1,1] to image(dtype=np.uint8) in range [0..255].

    Args:
        input_tensor: Input image tensor of Bx3xHxW layout, range [-1..1].
    Returns:
        A numpy image of layout BxHxWx3, range [0..255], uint8 dtype.
    """
    if range_min == -1:
        input_tensor = (input_tensor.float() + 1.0) / 2.0
    ndim = input_tensor.ndim
    output_image = input_tensor.clamp(0, 1).cpu().numpy()
    output_image = output_image.transpose((0,) + tuple(range(2, ndim)) + (1,))
    return (output_image * _UINT8_MAX_F + 0.5).astype(np.uint8)

def numpy2tensor(
    input_image: np.ndarray,
    dtype: torch.dtype = _DTYPE,
    device: str = _DEVICE,
    range_min: int = -1,
) -> torch.Tensor:
    """Converts image(dtype=np.uint8) to `dtype` in range [0..255].

    Args:
        input_image: A batch of images in range [0..255], BxHxWx3 layout.
    Returns:
        A torch.Tensor of layout Bx3xHxW in range [-1..1], dtype.
    """
    ndim = input_image.ndim
    indices = list(range(1, ndim))[-1:] + list(range(1, ndim))[:-1]
    image = input_image.transpose((0,) + tuple(indices)) / _UINT8_MAX_F
    if range_min == -1:
        image = 2.0 * image - 1.0
    return torch.from_numpy(image).to(dtype).to(device)


model_name = "Cosmos-0.1-Tokenizer-DV4x8x8"

import ipdb; ipdb.set_trace()
encoder = CausalVideoTokenizer(checkpoint_enc=f'pretrained_ckpts/{model_name}/encoder.jit')
decoder = CausalVideoTokenizer(checkpoint_dec=f'pretrained_ckpts/{model_name}/decoder.jit')



if False:
    input_tensor = torch.randn(1, 3, 9, 512, 512).to('cuda').to(torch.bfloat16)  # [B, C, T, H, W]
    # encoder._enc_model.quantizer.implicit_codebook.shape
    # import ipdb; ipdb.set_trace()
    (indices, codes) = encoder.encode(input_tensor)

    # The indices, Bx(t)x(h)x(w), from a codebook of size 64K,
    torch.testing.assert_close(indices.shape, (1, 3, 64, 64))

    # The discrete code, Bx6x(t)x(h)x(w), where the compression rate  is again (T/t x H/h x W/w), and channel dimension of 6.
    torch.testing.assert_close(codes.shape, (1, 6, 3, 64, 64))


    # 单张图片输入， 5D
    # input_tensor = torch.randn(1, 3, 1, 512, 512).to('cuda').to(torch.bfloat16)  # [B, C, T, H, W]
    # encoder = CausalVideoTokenizer(checkpoint_enc=f'pretrained_ckpts/{model_name}/encoder.jit')
    # (indices, codes) = encoder.encode(input_tensor) # torch.Size([1, 1, 64, 64])

    # The input tensor can be reconstructed by the decoder as:
    
    reconstructed_tensor = decoder.decode(indices)
    torch.testing.assert_close(reconstructed_tensor.shape, input_tensor.shape)

"""
import torch, os

# 加载 .jit 文件
model = torch.jit.load("/storage/zhubin/Janus-MoE/pretrained_ckpts/Cosmos-0.1-Tokenizer-DV4x8x8/encoder.jit")
def print_model_structure(module, prefix=""):
    for name, child in module.named_children():
        print(f"{prefix}{name}: {child}")
        print_model_structure(child, prefix + "  ")

print_model_structure(model._enc_model.quant_conv)


tokenizer_ckpt_path = os.path.join(checkpoint_dir, "Cosmos-1.0-Tokenizer-DV8x16x16/ema.jit")
self.image_processor = AutoImageProcessor.from_pretrained(image_processor_path)

"""

if True:

    # ===============
    filepath = '/storage/zhubin/Janus-MoE/useless_/3820-512.jpg'
    import mediapy as media 
    image = media.read_image(filepath) 
    image = np.expand_dims(image, axis=0) 
    image = numpy2tensor(image) #  BxHxWx3 layout. ->  Bx3xHxW 
    input_tensor = image

    input_tensor = input_tensor.unsqueeze(2)
    # import ipdb; ipdb.set_trace()
    
    """
    import torch
    jit_filepath= '/storage/zhubin/Janus-MoE/pretrained_ckpts/Cosmos-0.1-Tokenizer-DV4x8x8/encoder.jit'
    ckpts = torch.jit.load(jit_filepath, map_location='cuda')

    ckpts.encoder.norm_out.norm.bias
    tensor([-0.0903, -0.0693, -0.0547, -0.0830, -0.1128, -0.0579, -0.0884, -0.0742,
            -0.0874, -0.0908, -0.0972, -0.0850, -0.0972, -0.0625, -0.0554, -0.1099,
            -0.0262, -0.0408, -0.1338, -0.1006, -0.0859, -0.0972, -0.1123, -0.0364,

    model_dict['gen_vision_model._enc_model.encoder.norm_out.norm.bias']
    tensor([-0.0903, -0.0693, -0.0547, -0.0830, -0.1128, -0.0579, -0.0884, -0.0742,
            -0.0874, -0.0908, -0.0972, -0.0850, -0.0972, -0.0625, -0.0554, -0.1099,
            -0.0262, -0.0408, -0.1338, -0.1006, -0.0859, -0.0972, -0.1123, -0.0364,
    # 获取所有参数
    for name, param in ckpts.named_parameters():
        print(f"Parameter name: {name}")
        print(f"Parameter shape: {param.shape}")
        print(f"Parameter values: {param}")

    model_dict['gen_vision_model._enc_model.encoder.norm_out.norm.bias']


    ffmpeg -i /storage/zhubin/Janus-MoE/useless/3820.jpg   -vf "scale='min(512,iw)':-1"  /storage/zhubin/Janus-MoE/useless/3820-512.jpg
    """


    (indices, codes) = encoder.encode(input_tensor)
    
    # image = resize_image(image, short_size=args.short_size)
    # ===============

    reconstructed_tensor = decoder.decode(indices)
    reconstructed_tensor = reconstructed_tensor.squeeze(2) 

    # reconstructed_tensor = reconstructed_tensor.permute(1,2,0) 
    recon_image = tensor2numpy(reconstructed_tensor)[0]
    # torch.testing.assert_close(reconstructed_tensor.shape, input_tensor.shape)
    # print(reconstructed_tensor.shape, input_tensor.shape)
    media.write_image(filepath.replace('.jpg','_recon.jpg'), recon_image)

"""

conda activate janus_pro
cd /storage/zhubin/Janus-MoE/
python /storage/zhubin/Janus-MoE/test_cosmos.py


# Autoencoding videos using `Cosmos-DV` with a compression rate of 4x8x8.
model_name="Cosmos-0.1-Tokenizer-CV4x8x8"
python3 -m janus.models.cosmos_tokenizer.video_cli \
    --video_pattern '/storage/zhubin/Janus-MoE/useless/297638.mp4' \
    --mode=torch \
    --tokenizer_type=CV \
    --checkpoint_enc pretrained_ckpts/${model_name}/encoder.jit \
    --checkpoint_dec pretrained_ckpts/${model_name}/decoder.jit

    
# Autoencoding images using `Cosmos-DI` with a compression rate of 8x8.

cd /storage/zhubin/Janus-MoE/ 
source /storage/miniconda3/etc/profile.d/conda.sh
conda activate janus_pro
 
 
model_name="Cosmos-0.1-Tokenizer-CI8x8"
python3 -m janus.models.cosmos_tokenizer.image_cli \
    --image_pattern '/storage/zhubin/Janus-MoE/useless/20.jpg' \
    --mode=torch \
    --tokenizer_type=DI \
    --spatial_compression=8 \
    --checkpoint_enc pretrained_ckpts/${model_name}/encoder.jit \
    --checkpoint_dec pretrained_ckpts/${model_name}/decoder.jit



# Autoencoding images using `Cosmos-DI` with a compression rate of 8x8.
model_name="Cosmos-0.1-Tokenizer-DI8x8"
python3 -m janus.models.cosmos_tokenizer.image_cli \
    --image_pattern  '/storage/zhubin/Janus-MoE/useless/20.jpg'  \
    --mode=torch \
    --tokenizer_type=DI \
    --spatial_compression=8 \
    --checkpoint_enc pretrained_ckpts/${model_name}/encoder.jit \
    --checkpoint_dec pretrained_ckpts/${model_name}/decoder.jit

"""

