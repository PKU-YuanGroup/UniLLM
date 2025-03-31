
from janus.utils.io import load_pil_images
from janus.models import MultiModalityCausalLM, VLChatProcessor



model_path = "deepseek-ai/Janus-Pro-7B"
model_path = "/storage/jp/Janus/Janus-Pro-1B"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path, cache_dir = './cache_dir')

tokenizer = vl_chat_processor.tokenizer
input_ids = tokenizer.encode("Hello, how are you?", return_tensors="pt")
print(input_ids) # tensor([[100000,  17464,     11,    946,    418,    340,     30]])

# max_length是算上start_token的
input_ids = tokenizer.encode("Hello, how are you?", return_tensors="pt", max_length=5, truncation=True)
print(input_ids)  # tensor([[100000,  17464,     11,    946,    418]])


# pad是左填充
input_ids = tokenizer.encode("Hello, how are you?", return_tensors="pt", max_length=10, truncation=True, padding="max_length", )
print(input_ids)  # tensor([[100002, 100002, 100002, 100000,  17464,     11,    946,    418,    340,   30]]) 
import ipdb; ipdb.set_trace()

# decode_text = tokenizer.decode(input_ids[0][:8]); print(decode_text)
"""
ipdb> decode_text = tokenizer.decode(input_ids[0][:1]); print(decode_text)
<｜begin▁of▁sentence｜>
ipdb> decode_text = tokenizer.decode(input_ids[0][:2]); print(decode_text)
<｜begin▁of▁sentence｜>Hello
ipdb> decode_text = tokenizer.decode(input_ids[0][:3]); print(decode_text)
<｜begin▁of▁sentence｜>Hello,
ipdb> decode_text = tokenizer.decode(input_ids[0][:4]); print(decode_text)
<｜begin▁of▁sentence｜>Hello, how
ipdb> decode_text = tokenizer.decode(input_ids[0][:5]); print(decode_text)
<｜begin▁of▁sentence｜>Hello, how are
ipdb> decode_text = tokenizer.decode(input_ids[0][:6]); print(decode_text)
<｜begin▁of▁sentence｜>Hello, how are you
ipdb> decode_text = tokenizer.decode(input_ids[0][:7]); print(decode_text)
<｜begin▁of▁sentence｜>Hello, how are you?
ipdb> decode_text = tokenizer.decode(input_ids[0][:8]); print(decode_text)
<｜begin▁of▁sentence｜>Hello, how are you?

"""