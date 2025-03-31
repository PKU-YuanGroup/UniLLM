import torch
# import torch_npu
# from torch_npu.contrib import transfer_to_npu
from janus.models.cosmos_tokenizer.video_lib import CausalVideoTokenizer
from janus.models.cosmos_tokenizer.networks import TokenizerConfigs
import  numpy as np
import os

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



mode =  "torch"
tokenizer_type  = "DV"
spatial_compression = 8
temporal_compression = 4
if  mode == "torch":
        tokenizer_config = TokenizerConfigs[tokenizer_type].value
        tokenizer_config.update(dict(spatial_compression=spatial_compression))
        tokenizer_config.update(dict(temporal_compression=temporal_compression))
else:
    tokenizer_config = None

model_name = "Cosmos-0.1-Tokenizer-DV4x8x8"
# input_tensor = torch.randn(1, 3, 9, 512, 512).to('npu').to(torch.bfloat16)  # [B, C, T, H, W]
filepath = '/storage/zhubin/Janus-MoE/output.png'
import mediapy as media 
image = media.read_image(filepath) 
image = np.expand_dims(image, axis=0) 
image = numpy2tensor(image) #  BxHxWx3 layout. ->  Bx3xHxW 
input_tensor = image.to(torch.bfloat16)

input_tensor = input_tensor.unsqueeze(2)

import ipdb; ipdb.set_trace()
# /storage/zhubin/Janus-MoE/pretrained_ckpts/Cosmos-0.1-Tokenizer-DV4x8x8/decoder.jit
assert os.path.exists(f'/storage/zhubin/Janus-MoE/pretrained_ckpts/{model_name}/encoder.jit')
assert os.path.exists(f'/storage/zhubin/Janus-MoE/pretrained_ckpts/{model_name}/decoder.jit')
encoder = CausalVideoTokenizer(checkpoint_enc=f'/storage/zhubin/Janus-MoE/pretrained_ckpts/{model_name}/encoder.jit', tokenizer_config=tokenizer_config)
decoder = CausalVideoTokenizer(checkpoint_dec=f'/storage/zhubin/Janus-MoE/pretrained_ckpts/{model_name}/decoder.jit', tokenizer_config=tokenizer_config)

(indices, codes) = encoder.encode(input_tensor)
# torch.testing.assert_close(indices.shape, (1, 3, 64, 64))
# torch.testing.assert_close(codes.shape, (1, 6, 3, 64, 64))
reconstructed_tensor = decoder.decode(indices)
reconstructed_tensor = reconstructed_tensor.squeeze(2) 
# reconstructed_tensor = reconstructed_tensor.permute(1,2,0) 
recon_image = tensor2numpy(reconstructed_tensor)[0]
# torch.testing.assert_close(reconstructed_tensor.shape, input_tensor.shape)
# print(reconstructed_tensor.shape, input_tensor.shape)
media.write_image(filepath.replace('.png','_recon.png'), recon_image)




# # The input tensor can be reconstructed by the decoder as:
# decoder = CausalVideoTokenizer(checkpoint_dec=f'../pretrained_ckpts/{model_name}/decoder.jit', tokenizer_config=continuous_video)
# reconstructed_tensor = decoder.decode(indices)
# torch.testing.assert_close(reconstructed_tensor.shape, input_tensor.shape)

# import torch
# import torch_npu
# from torch_npu.contrib import transfer_to_npu
# from cosmos_tokenizer.video_lib import CausalVideoTokenizer
# from cosmos_tokenizer.networks.configs import continuous_video

# model_name = "Cosmos-Tokenizer-CV4x8x8"
# input_tensor = torch.randn(1, 3, 9, 512, 512).to('npu').to(torch.bfloat16)  # [B, C, T, H, W]
# encoder = CausalVideoTokenizer(checkpoint_enc=f'pretrained_ckpts/{model_name}/encoder.jit', tokenizer_config=continuous_video)
# (latent,) = encoder.encode(input_tensor)
# torch.testing.assert_close(latent.shape, (1, 16, 2, 64, 64))

