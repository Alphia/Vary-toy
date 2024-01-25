import os
import warnings
from io import BytesIO

import requests
import torch
from PIL import Image
from transformers import AutoTokenizer
from transformers import CLIPImageProcessor
from transformers import TextStreamer

from vary.model import *
from vary.model.plug.blip_process import BlipImageEvalProcessor
from vary.utils.conversation import conv_templates, SeparatorStyle
from vary.utils.utils import KeywordsStoppingCriteria
from vary.utils.utils import disable_torch_init

warnings.filterwarnings("ignore", category=UserWarning, module='torch.nn.modules.module')

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = '<imgpad>'
DEFAULT_IM_START_TOKEN = '<img>'
DEFAULT_IM_END_TOKEN = '</img>'
MODEL_NAME = "/cache/Vary-toy/"
CONV_MODE = "mpt"


def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def init_model():
    # Model
    disable_torch_init()
    model_name = os.path.expanduser(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = varyQwenForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, device_map='cuda',
                                                trust_remote_code=True)
    model.to(device='cuda', dtype=torch.bfloat16)
    image_processor = CLIPImageProcessor.from_pretrained(
        "/cache/vit-large-patch14/vit-large-patch14/", torch_dtype=torch.float16)
    return tokenizer, model, image_processor


tokenizer, model, image_processor = init_model()


def eval_model(image_url, qs):
    image_token_len = 256
    formatted_qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len + DEFAULT_IM_END_TOKEN + '\n' + qs
    conv = conv_templates[CONV_MODE].copy()
    conv.append_message(conv.roles[0], formatted_qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    inputs = tokenizer([prompt])

    image = load_image(image_url)
    image_processor_high = BlipImageEvalProcessor(image_size=1024)
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
    image_tensor_2 = image_processor_high(image.copy())

    input_ids = torch.as_tensor(inputs.input_ids).cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        output_ids = model.generate(
            input_ids,
            images=[(image_tensor.unsqueeze(0).half().cuda(), image_tensor_2.unsqueeze(0).half().cuda())],
            do_sample=True,
            num_beams=1,
            # temperature=0.2,
            streamer=streamer,
            max_new_tokens=2048,
            stopping_criteria=[stopping_criteria]
        )
        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()

        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        return outputs.strip().replace('<|im_end|>', '')
