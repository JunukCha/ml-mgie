import os

from tqdm.auto import tqdm

from PIL import Image

import torch as T
import transformers, diffusers
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from CustomDataset import CustomDatasetHQ_EVAL
from llava.conversation import conv_templates
from llava.model import *
import numpy as np
import numpy as np

import lpips
from torchvision.transforms import ToTensor, Resize, Compose
from skimage.metrics import structural_similarity as ssim
import torch

def calculate_ssim(img1, img2):
    data1 = np.array(img1)
    data2 = np.array(img2)
    # data1 = cv2.resize(data1, (256, 256))
    # data2 = cv2.resize(data2, (256, 256))
    ssim_value = ssim(data1, data2, data_range=data1.max() - data1.min(), multichannel=True, channel_axis=2)
    return ssim_value

def calculate_psnr(img1, img2):
    mse = np.mean((np.array(img1) - np.array(img2)) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

def calculate_lpips(img1, img2):
    transform = Compose([
        Resize((256, 256)), 
        ToTensor() 
    ])

    img1 = transform(img1.convert('RGB')).unsqueeze(0)
    img2 = transform(img2.convert('RGB')).unsqueeze(0)

    loss_fn = lpips.LPIPS(net='vgg').eval()

    with torch.no_grad():
        distance = loss_fn(img1, img2)

    return distance.item()

def concatenate_horizontally_pil(images, padding_size=0, padding_color=(255, 255, 255)):
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths) + padding_size * (len(images) - 1)
    max_height = max(heights)

    new_img = Image.new("RGB", (total_width, max_height), color=padding_color)

    x_offset = 0
    for img in images:
        new_img.paste(img, (x_offset, 0))
        x_offset += img.width + padding_size
    
    return new_img

def crop_resize(f, sz=512):
    w, h = f.size
    if w>h:
        p = (w-h)//2
        f = f.crop([p, 0, p+h, h])
    elif h>w:
        p = (h-w)//2
        f = f.crop([0, p, w, p+w])
    f = f.resize([sz, sz])
    return f
def remove_alter(s):  # hack expressive instruction
    if 'ASSISTANT:' in s: s = s[s.index('ASSISTANT:')+10:].strip()
    if '</s>' in s: s = s[:s.index('</s>')].strip()
    if 'alternative' in s.lower(): s = s[:s.lower().index('alternative')]
    if '[IMG0]' in s: s = s[:s.index('[IMG0]')]
    s = '.'.join([s.strip() for s in s.split('.')[:2]])
    if s[-1]!='.': s += '.'
    return s.strip()


def main():
    DEFAULT_IMAGE_TOKEN = '<image>'
    DEFAULT_IMAGE_PATCH_TOKEN = '<im_patch>'
    DEFAULT_IM_START_TOKEN = '<im_start>'
    DEFAULT_IM_END_TOKEN = '<im_end>'
    PATH_LLAVA = './_ckpt/LLaVA-7B-v1'

    tokenizer = transformers.AutoTokenizer.from_pretrained(PATH_LLAVA)
    model = LlavaLlamaForCausalLM.from_pretrained(PATH_LLAVA, low_cpu_mem_usage=True, torch_dtype=T.float16, use_cache=True).cuda()
    image_processor = transformers.CLIPImageProcessor.from_pretrained(model.config.mm_vision_tower, torch_dtype=T.float16)

    tokenizer.padding_side = 'left'
    tokenizer.add_tokens(['[IMG0]', '[IMG1]', '[IMG2]', '[IMG3]', '[IMG4]', '[IMG5]', '[IMG6]', '[IMG7]'], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))
    ckpt = T.load('./_ckpt/mgie_7b/mllm.pt', map_location='cpu')
    model.load_state_dict(ckpt, strict=False)

    mm_use_im_start_end = getattr(model.config, 'mm_use_im_start_end', False)
    tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end: tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

    vision_tower = model.get_model().vision_tower[0]
    vision_tower = transformers.CLIPVisionModel.from_pretrained(vision_tower.config._name_or_path, torch_dtype=T.float16, low_cpu_mem_usage=True).cuda()
    model.get_model().vision_tower[0] = vision_tower
    vision_config = vision_tower.config
    vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
    vision_config.use_im_start_end = mm_use_im_start_end
    if mm_use_im_start_end: vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])
    image_token_len = (vision_config.image_size//vision_config.patch_size)**2

    _ = model.eval()
    EMB = ckpt['emb'].cuda()
    with T.inference_mode(): NULL = model.edit_head(T.zeros(1, 8, 4096).half().to('cuda'), EMB)
    print('NULL:', NULL.shape)

    pipe = diffusers.StableDiffusionInstructPix2PixPipeline.from_pretrained('timbrooks/instruct-pix2pix', torch_dtype=T.float16, safety_checker=None).to('cuda')
    pipe.set_progress_bar_config(disable=True)
    pipe.unet.load_state_dict(T.load('./_ckpt/mgie_7b/unet.pt', map_location='cpu'))

    SEED = 13331

    num_inference_steps = 25
    image_guidance_scale = 1.0
    guidance_scale = 7
    
    dataset = CustomDatasetHQ_EVAL()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    psnr_list = []
    ssim_list = []
    lpips_list = []
    for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        img_x, target, txt = data
        target = Image.fromarray(target.cpu().numpy()[0])
        prompt = txt[0]
        
        img = image_processor.preprocess(img_x, return_tensors='pt')['pixel_values'][0]
        txt = "what will this image be like if '%s'"%(prompt)
        txt = txt+'\n'+DEFAULT_IM_START_TOKEN+DEFAULT_IMAGE_PATCH_TOKEN*image_token_len+DEFAULT_IM_END_TOKEN
        conv = conv_templates['vicuna_v1_1'].copy()
        conv.append_message(conv.roles[0], txt), conv.append_message(conv.roles[1], None)
        txt = conv.get_prompt()
        txt = tokenizer(txt)
        txt, mask = T.as_tensor(txt['input_ids']), T.as_tensor(txt['attention_mask'])
        
        with T.inference_mode():
            out = model.generate(txt.unsqueeze(dim=0).cuda(), images=img.half().unsqueeze(dim=0).cuda(), attention_mask=mask.unsqueeze(dim=0).cuda(), 
                                do_sample=False, max_new_tokens=96, num_beams=1, no_repeat_ngram_size=3, 
                                return_dict_in_generate=True, output_hidden_states=True)
            out, hid = out['sequences'][0].tolist(), T.cat([x[-1] for x in out['hidden_states']], dim=1)[0]
            
            p = min(out.index(32003)-1 if 32003 in out else len(hid)-9, len(hid)-9)
            hid = hid[p:p+8]

            out = remove_alter(tokenizer.decode(out))
            emb = model.edit_head(hid.unsqueeze(dim=0), EMB)
            res = pipe(
                image=img_x, prompt_embeds=emb, 
                negative_prompt_embeds=NULL, 
                generator=T.Generator(device='cuda').manual_seed(SEED), 
                num_inference_steps=num_inference_steps,
                image_guidance_scale=image_guidance_scale,
                guidance_scale=guidance_scale,
            ).images[0]

        results_folder = f"eval_outputs_HQ/{i:03d}"
        os.makedirs(results_folder, exist_ok=True)
        img_x = Image.fromarray((img_x[0].permute(1, 2, 0).cpu().numpy()*255).astype(np.uint8))
        img_x.save(os.path.join(results_folder, "input.jpg"))
        res.save(os.path.join(results_folder, "pred.jpg"))
        target.save(os.path.join(results_folder, "target.jpg"))
        
        with open(os.path.join(results_folder, "text.txt"), "w") as f:
            f.write(prompt)
        
        concatenated_image = concatenate_horizontally_pil([img_x, res], padding_size=10)
        concatenated_image.save(os.path.join(results_folder, "concatenated_image.jpg"))
        
        psnr = calculate_psnr(target, res)
        psnr_list.append(psnr)
        ssim = calculate_ssim(target, res)
        ssim_list.append(ssim)
        lpips = calculate_lpips(target, res)
        lpips_list.append(lpips)
    
    print("psnr", np.array(psnr_list).mean())
    print("ssim", np.array(ssim_list).mean())
    print("lpips", np.array(lpips_list).mean())
    with open("eval_outputs_HQ/evaluate.txt", "w") as f:
        f.write(f"psnr: {np.array(psnr_list).mean():.4f}\n")
        f.write(f"ssim: {np.array(ssim_list).mean():.4f}\n")
        f.write(f"lpips: {np.array(lpips_list).mean():.4f}")

if __name__ == "__main__":
    main()