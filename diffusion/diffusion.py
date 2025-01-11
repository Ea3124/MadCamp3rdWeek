# Standard library imports
import math
import os
import random

# Third-party imports
import numpy as np
import pandas
import glob
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from PIL import Image
from accelerate import Accelerator
from accelerate.logging import get_logger
from datasets import load_dataset
from bing_image_downloader import downloader
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from tqdm.auto import tqdm
from torchvision import transforms
from transformers import (
    AutoTokenizer,
    PretrainedConfig,
    BlipProcessor,
    BlipForConditionalGeneration,
)

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
    StableVideoDiffusionPipeline,
    StableDiffusionPipeline
)
from diffusers.loaders import LoraLoaderMixin
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params
from diffusers.utils import (
    make_image_grid,
    convert_state_dict_to_diffusers,
    export_to_video,
    export_to_gif,
)
from diffusers.utils.torch_utils import is_compiled_module
if torch.cuda.is_available():
    print(f"GPU is available: {torch.cuda.get_device_name(0)}")
else:
    print("GPU is not available.")
# custom_dataset_folder_name = "gen_ai_custom_dataset"
# if not os.path.exists(custom_dataset_folder_name):
#     os.makedirs(custom_dataset_folder_name)
# if not os.path.exists(os.path.join(custom_dataset_folder_name, "train")):
#     os.makedirs(os.path.join(custom_dataset_folder_name, "train"))
# custom_dataset_folder_path = os.path.join(os.getcwd(), custom_dataset_folder_name)

# query = "Plaster bust of a human face photograph"
# downloader.download(
#     query,
#     limit=100,
#     output_dir=custom_dataset_folder_path,
#     adult_filter_off=True,
#     force_replace=False,
#     timeout=60,
#     verbose=False,
# )

# # 파일 이름 변경
# for idx, item in enumerate(
#     os.listdir(
#         os.path.join(
#             custom_dataset_folder_path,
#             query,
#         )
#     )
# ):
#     format = item.split(".")[-1]
#     # idx format 0 -> 00
#     idx = str(idx).zfill(2)
#     os.rename(
#         os.path.join(custom_dataset_folder_path, query, item),
#         os.path.join(custom_dataset_folder_path, "train", f"{idx}.{format}"),
#     )

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
# model = BlipForConditionalGeneration.from_pretrained(
#     "Salesforce/blip-image-captioning-base", torch_dtype=torch.float16
# ).to(device)

# test_image = Image.open(next(iter(glob.glob(os.path.join(custom_dataset_folder_path, "train", "87*"))), None))
# test_image

# inputs = processor(test_image, return_tensors="pt").to(device, torch.float16)
# out = model.generate(**inputs, max_new_tokens=50)
# print(processor.decode(out[0], skip_special_tokens=True))

# metadata = pandas.DataFrame(columns=["file_name", "caption"])

# for idx, item in enumerate(
#     os.listdir(os.path.join(custom_dataset_folder_path, "train"))
# ):
#     format = item.split(".")[-1]
#     image_path = os.path.join(custom_dataset_folder_path, "train", item)
#     test_image = Image.open(image_path)
#     inputs = processor(test_image, return_tensors="pt").to(device, torch.float16)
#     out = model.generate(**inputs, max_new_tokens=50)
#     caption = processor.decode(out[0], skip_special_tokens=True)
#     metadata.loc[idx] = [item, caption]

# # sort by Image column
# metadata = metadata.sort_values(by=["file_name"]).reset_index(drop=True)
# metadata.head(5)

# # save metadata
# metadata.to_csv(
#     os.path.join(custom_dataset_folder_path, "train", "metadata.csv"), index=False
# )

# custom_dataset = load_dataset("imagefolder", data_dir=custom_dataset_folder_path)
# custom_dataset["train"][0]

# del model
# torch.cuda.empty_cache()

# def tokenize_prompt(tokenizer, prompt):
#     text_inputs = tokenizer(
#         prompt,
#         padding="max_length",
#         max_length=tokenizer.model_max_length,
#         truncation=True,
#         return_tensors="pt",
# )
#     text_input_ids = text_inputs.input_ids
#     return text_input_ids


# # Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
# def encode_prompt(text_encoders, tokenizers, prompt, text_input_ids_list=None):
#     prompt_embeds_list = []

#     for i, text_encoder in enumerate(text_encoders):
#         if tokenizers is not None:
#             tokenizer = tokenizers[i]
#             text_input_ids = tokenize_prompt(tokenizer, prompt)
#         else:
#             assert text_input_ids_list is not None
#             try:
#                 text_input_ids = text_input_ids_list[i]
#             except IndexError:
#                 pass
#             #text_input_ids = text_input_ids_list[i]

#         prompt_embeds = text_encoder(
#             text_input_ids.to(text_encoder.device),
#             output_hidden_states=True,
#             return_dict=False,
#     )

#         # We are only ALWAYS interested in the pooled output of the final text encoder
#         pooled_prompt_embeds = prompt_embeds[0]
#         prompt_embeds = prompt_embeds[-1][-2]
#         bs_embed, seq_len, _ = prompt_embeds.shape
#         prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
#         prompt_embeds_list.append(prompt_embeds)

#     prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
#     pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
#     return prompt_embeds, pooled_prompt_embeds

# def import_model_class_from_model_name_or_path(
#     pretrained_model_name_or_path: str, subfolder: str = "text_encoder"
# ):
#     text_encoder_config = PretrainedConfig.from_pretrained(
#         pretrained_model_name_or_path, subfolder=subfolder
# )
#     model_class = text_encoder_config.architectures[0]

#     if model_class == "CLIPTextModel":
#         from transformers import CLIPTextModel

#         return CLIPTextModel(text_encoder_config).from_pretrained(
#             pretrained_model_name_or_path, subfolder=subfolder
#         )
#     elif model_class == "CLIPTextModelWithProjection":
#         from transformers import CLIPTextModelWithProjection

#         return CLIPTextModelWithProjection(text_encoder_config).from_pretrained(
#             pretrained_model_name_or_path, subfolder=subfolder
#         )
#     else:
#         raise ValueError(f"{model_class} is not supported.")
    
# MODEL_NAME  = "runwayml/stable-diffusion-v1-5"
# VAE_NAME ="stabilityai/sd-vae-ft-mse"
# VARIANT = "fp16"

# MODEL_NAME  = "stabilityai/stable-diffusion-xl-base-1.0"
# VAE_NAME ="madebyollin/sdxl-vae-fp16-fix"
# VARIANT = "fp16"

# tokenizer_one = AutoTokenizer.from_pretrained(
#         pretrained_model_name_or_path = MODEL_NAME,
#         subfolder = "tokenizer",
#         use_fast = False,
# )
# text_encoder_one = import_model_class_from_model_name_or_path(
#         pretrained_model_name_or_path = MODEL_NAME,
#         subfolder = "text_encoder",
# )

# tokenizer_two =  None
# text_encoder_two = None

# # StableDiffusionXL의 경우 두개의 Text Encoder를 사용하여 두개의 Tokenizer를 불러옵니다.
# tokenizer_two = AutoTokenizer.from_pretrained(
#         pretrained_model_name_or_path = MODEL_NAME,
#         subfolder = "tokenizer_2",
#         use_fast = False,
# )


# text_encoder_two = import_model_class_from_model_name_or_path(
#         pretrained_model_name_or_path = MODEL_NAME,
#         subfolder = "text_encoder_2",
# )

# noise_scheduler = DDPMScheduler.from_pretrained(
#         pretrained_model_name_or_path = MODEL_NAME,
#         subfolder = "scheduler"
# )

# vae = AutoencoderKL.from_pretrained(
#     pretrained_model_name_or_path = VAE_NAME,
# )

# unet = UNet2DConditionModel.from_pretrained(
#         pretrained_model_name_or_path = MODEL_NAME,
#         subfolder = "unet",
#         variant = VARIANT,
# )

# vae.requires_grad_(False)
# text_encoder_one.requires_grad_(False)
# unet.requires_grad_(False)
# print("VAE, Text Encoder, and UNet are frozen.")

# # StableDiffusionXL의 경우 두개의 Text Encoder를 사용하여 두개의 Tokenizer를 불러옵니다.
# text_encoder_two.requires_grad_(False)

# seed = 2015
# train_batch_size = 2
# resolution = 1024
# max_train_steps = 10000
# checkpointing_steps = 5000
# learning_rate = 1e-4
# output_dir = "./sd_lora"

# # Make sure output_dir exists
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# logger = get_logger(__name__)

# def training_function(dataset, unet, vae, text_encoder_one, text_encoder_two=None):

#     # Init Accelerator
#     accelerator = Accelerator(mixed_precision=VARIANT)
#     weight_dtype = torch.float32
#     if accelerator.mixed_precision == "fp16":
#         weight_dtype = torch.float16
#     elif accelerator.mixed_precision == "bf16":
#         weight_dtype = torch.bfloat16

#     # 세부 모델들 GPU로 올리기, 및 datatype 설정
#     unet.to(accelerator.device, dtype=weight_dtype)
#     vae.to(accelerator.device, dtype=weight_dtype)
#     text_encoder_one.to(accelerator.device, dtype=weight_dtype)
#     if text_encoder_two is not None:
#         text_encoder_two.to(accelerator.device, dtype=weight_dtype)

#     # LoRA config 설정
#     # UNet의 Attention Layer에 LoRA weights를 추가
#     # Set correct lora layers
#     unet_lora_config = LoraConfig(
#         r=4, # Dimension of LoRA update Matrices
#         lora_alpha=4, # Dimension of LoRA update Matrices
#         init_lora_weights="gaussian",
#         target_modules=["to_k", "to_q", "to_v", "to_out.0"],
#     )
#     unet.add_adapter(unet_lora_config)

#     # Trainable Parameter (LoRA weights) 만을 FP32로 Upcast
#     cast_training_params([unet], dtype=torch.float32)

#     # Accelerator에 wrap된 모델을 unwrap하는 내부함수
#     def unwrap_model(model):
#         model = accelerator.unwrap_model(model)
#         model = model._orig_mod if is_compiled_module(model) else model
#         return model


#     # Custom 된  save & loading hook 함수. LoRA Weight만을 저장하고 불러오는 용도로 사용
#     # 실질적으로 accelrator.save_state() 함수 사용을 위해 필요한 함수
#     def save_model_hook(models, weights, output_dir):
#         if accelerator.is_main_process:
#             # there are only two options here. Either are just the unet attn processor layers
#             # or there are the unet and text encoder atten layers
#             unet_lora_layers_to_save = None
#             text_encoder_one_lora_layers_to_save = None
#             if text_encoder_two is not None:
#                 text_encoder_two_lora_layers_to_save = None

#             for model in models:
#                 if isinstance(unwrap_model(model), type(unwrap_model(unet))):
#                     unet_lora_layers_to_save = convert_state_dict_to_diffusers(
#                         get_peft_model_state_dict(model)
#                     )
#                 elif isinstance(
#                     unwrap_model(model), type(unwrap_model(text_encoder_one))
#                 ):
#                     text_encoder_one_lora_layers_to_save = (
#                         convert_state_dict_to_diffusers(
#                             get_peft_model_state_dict(model)
#                         )
#                     )
#                 elif isinstance(
#                     unwrap_model(model), type(unwrap_model(text_encoder_two))
#                 ):
#                     text_encoder_two_lora_layers_to_save = (
#                         convert_state_dict_to_diffusers(
#                             get_peft_model_state_dict(model)
#                         )
#                     )
#                 else:
#                     raise ValueError(f"unexpected save model: {model.__class__}")

#                 # make sure to pop weight so that corresponding model is not saved again
#                 if weights:
#                     weights.pop()

#             if text_encoder_two is not None:
#                 StableDiffusionXLPipeline.save_lora_weights(
#                     output_dir,
#                     unet_lora_layers=unet_lora_layers_to_save,
#                     text_encoder_lora_layers=text_encoder_one_lora_layers_to_save,
#                     text_encoder_2_lora_layers=text_encoder_two_lora_layers_to_save,
#                 )
#             else:
#                 StableDiffsuionPipeline.save_lora_weights(
#                     output_dir,
#                     unet_lora_layers=unet_lora_layers_to_save,
#                     text_encoder_lora_layers=text_encoder_one_lora_layers_to_save,
#                 )

#     def load_model_hook(models, input_dir):
#         unet_ = None
#         text_encoder_one_ = None
#         text_encoder_two_ = None

#         while len(models) > 0:
#             model = models.pop()

#             if isinstance(model, type(unwrap_model(unet))):
#                 unet_ = model
#             elif isinstance(model, type(unwrap_model(text_encoder_one))):
#                 text_encoder_one_ = model
#             elif isinstance(model, type(unwrap_model(text_encoder_two))):
#                 text_encoder_two_ = model
#             else:
#                 raise ValueError(f"unexpected save model: {model.__class__}")

#         lora_state_dict, network_alphas = LoraLoaderMixin.lora_state_dict(input_dir)
#         LoraLoaderMixin.load_lora_into_unet(
#             lora_state_dict, network_alphas=network_alphas, unet=unet_
#         )

#         text_encoder_state_dict = {
#             k: v for k, v in lora_state_dict.items() if "text_encoder." in k
#         }
#         LoraLoaderMixin.load_lora_into_text_encoder(
#             text_encoder_state_dict,
#             network_alphas=network_alphas,
#             text_encoder=text_encoder_one_,
#         )
#         if text_encoder_two_ is not None:
#             text_encoder_2_state_dict = {
#                 k: v for k, v in lora_state_dict.items() if "text_encoder_2." in k
#             }
#             LoraLoaderMixin.load_lora_into_text_encoder(
#                 text_encoder_2_state_dict,
#                 network_alphas = network_alphas,
#                 text_encoder = text_encoder_two_,
#             )

#     accelerator.register_save_state_pre_hook(save_model_hook)
#     accelerator.register_load_state_pre_hook(load_model_hook)

#     # Optimizer 설정
#     optimizer_class = torch.optim.AdamW
#     # Optimizer creation
#     params_to_optimize = list(filter(lambda p: p.requires_grad, unet.parameters()))
#     optimizer = optimizer_class(
#         params_to_optimize,
#         lr=learning_rate*train_batch_size,
#         betas = (0.9, 0.999),
#         weight_decay = 1e-2,
#         eps = 1e-08,
#     )

#     # Preprocessing the datasets
#     # 먼저 데이터셋의 Column 구분 (Image, Caption)
#     column_names = dataset["train"].column_names
#     image_column = column_names[0]
#     caption_column = column_names[1]

#    # Captipon 들을 두가지 Text Tokenizer로 Tokenize 하는 내부함수 정의
#     def tokenize_captions(examples, is_train=True,):
#         captions = []
#         for caption in examples[caption_column]:
#             if isinstance(caption, str):
#                 captions.append(caption)
#             elif isinstance(caption, (list, np.ndarray)):
#                 # take a random caption if there are multiple
#                 captions.append(random.choice(caption) if is_train else caption[0])
#             else:
#                 raise ValueError(
#                     f"Caption column `{caption_column}` should contain either strings or lists of strings."
#                 )
#         tokens_one = tokenize_prompt(tokenizer_one, captions)
#         if text_encoder_two is not None:
#             tokens_two = tokenize_prompt(tokenizer_two, captions)
#         else:
#             tokens_two = None
#         return tokens_one, tokens_two

#     # Image 를 전처리 하는 부분 정의
#     train_resize = transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR)
#     train_crop = transforms.CenterCrop(resolution)
#     train_flip = transforms.RandomHorizontalFlip(p=1.0)
#     train_transforms = transforms.Compose(
#         [
#             transforms.ToTensor(),
#             transforms.Normalize([0.5], [0.5]),
#         ]
#     )

#     # Image 와 Caption preprocessing 을 위한 종합 내부함수정의
#     def preprocess_train(examples):
#         images = [image.convert("RGB") for image in examples[image_column]]
#         # image aug
#         original_sizes = []
#         all_images = []
#         crop_top_lefts = []
#         for image in images:
#             original_sizes.append((image.height, image.width))
#             image = train_resize(image)
#             if random.random() < 0.5:# flip
#                 image = train_flip(image)
#             y1 = max(0, int(round((image.height - resolution) / 2.0)))
#             x1 = max(0, int(round((image.width - resolution) / 2.0)))
#             image = train_crop(image)
#             crop_top_left = (y1, x1)
#             crop_top_lefts.append(crop_top_left)
#             image = train_transforms(image)
#             all_images.append(image)

#         examples["original_sizes"] = original_sizes
#         examples["crop_top_lefts"] = crop_top_lefts
#         examples["pixel_values"] = all_images
#         tokens_one, tokens_two = tokenize_captions(examples)
#         examples["input_ids_one"] = tokens_one
#         if text_encoder_two is not None:
#             examples["input_ids_two"] = tokens_two
#         return examples

#     # Training Dataset 생성
#     train_dataset = dataset["train"].with_transform(preprocess_train)

#     # DataLoader 생성
#     def collate_fn(examples):
#         pixel_values = torch.stack([example["pixel_values"] for example in examples])
#         pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
#         original_sizes = [example["original_sizes"] for example in examples]
#         crop_top_lefts = [example["crop_top_lefts"] for example in examples]
#         input_ids_one = torch.stack([example["input_ids_one"] for example in examples])
#         if text_encoder_two is not None:
#             input_ids_two = torch.stack([example["input_ids_two"] for example in examples])
#             return {
#                 "pixel_values": pixel_values,
#                 "input_ids_one": input_ids_one,
#                 "input_ids_two": input_ids_two,
#                 "original_sizes": original_sizes,
#                 "crop_top_lefts": crop_top_lefts,
#             }
#         else:
#             return {
#                 "pixel_values": pixel_values,
#                 "input_ids_one": input_ids_one,
#                 "original_sizes": original_sizes,
#                 "crop_top_lefts": crop_top_lefts,
#             }
#     train_dataloader = torch.utils.data.DataLoader(
#         train_dataset,
#         shuffle = True,
#         collate_fn = collate_fn,
#         batch_size = train_batch_size,
#         num_workers = 0,
#     )

#     lr_scheduler = get_scheduler(
#         'cosine',
#         optimizer=optimizer,
#         num_warmup_steps = 500 * 1,
#         num_training_steps = max_train_steps *1,
#     )

#     # Prepare with  accelerator
#     unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(unet, optimizer, train_dataloader, lr_scheduler)


#     # Training 시작 전 기본 세팅
#     num_train_epochs = math.ceil(max_train_steps / len(train_dataset))
#     total_batch_size = train_batch_size * accelerator.num_processes
#     logger.info("***** Running training *****")
#     logger.info(f"  Num examples = {len(train_dataset)}")
#     logger.info(f"  Num Epochs = {num_train_epochs}")
#     logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
#     logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
#     logger.info(f"  Total optimization steps = {max_train_steps}")
#     global_step = 0
#     first_epoch = 0
#     initial_global_step = 0

#     progress_bar = tqdm(
#         range(0, max_train_steps),
#         initial = initial_global_step,
#         desc="Steps",
#         # Only show the progress bar once on each machine.
#         disable = not accelerator.is_local_main_process,
#     )

#     # Training Loop 시작
#     for epoch in range(first_epoch, num_train_epochs):
#         unet.train()
#         train_loss = 0.0
#         for step, batch in enumerate(train_dataloader):
#             with accelerator.accumulate(unet):
#                 # Convert images to latent space
#                 pixel_values = batch["pixel_values"].to(dtype=weight_dtype)
#                 model_input = vae.encode(pixel_values).latent_dist.sample()
#                 model_input = model_input * vae.config.scaling_factor

#                 # Sample noise that we'll add to the latents
#                 noise = torch.randn_like(model_input)

#                 bsz = model_input.shape[0]
#                 # Sample a random timestep for each image
#                 timesteps = torch.randint(
#                     0,
#                     noise_scheduler.config.num_train_timesteps,
#                     (bsz,),
#                     device=model_input.device,
#                 )
#                 timesteps = timesteps.long()

#                 # Add noise to the model input according to the noise magnitude at each timestep
#                 # (this is the forward diffusion process)
#                 noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)

#                 # time ids
#                 def compute_time_ids(original_size, crops_coords_top_left):
#                     # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
#                     target_size = (resolution, resolution)
#                     add_time_ids = list(original_size + crops_coords_top_left + target_size)
#                     add_time_ids = torch.tensor([add_time_ids])
#                     add_time_ids = add_time_ids.to(accelerator.device, dtype=weight_dtype)
#                     return add_time_ids

#                 add_time_ids = torch.cat(
#                     [
#                         compute_time_ids(s, c)
#                         for s, c in zip(
#                             batch["original_sizes"], batch["crop_top_lefts"]
#                         )
#                     ]
#                 )

#                 # Predict the noise residual
#                 unet_added_conditions = {"time_ids": add_time_ids}

#                 if text_encoder_two is not None:
#                     text_input_ids_list = [
#                         batch["input_ids_one"],
#                         batch["input_ids_two"],
#                     ]
#                     prompt_embeds, pooled_prompt_embeds = encode_prompt(
#                         text_encoders=[text_encoder_one, text_encoder_two],
#                         tokenizers=None,
#                         prompt=None,
#                         text_input_ids_list=text_input_ids_list,
#                     )
#                 else:
#                     text_input_ids_list = [batch["input_ids_one"]]

#                     prompt_embeds, pooled_prompt_embeds = encode_prompt(
#                         text_encoders=[text_encoder_one],
#                         tokenizers=None,
#                         prompt=None,
#                         text_input_ids_list=text_input_ids_list,
#                     )
#                 unet_added_conditions.update({"text_embeds": pooled_prompt_embeds})
#                 model_pred = unet(
#                     noisy_model_input,
#                     timesteps,
#                     prompt_embeds,
#                     added_cond_kwargs=unet_added_conditions,
#                     return_dict=False,
#                 )[0]


#                 if noise_scheduler.config.prediction_type == "epsilon":
#                     target = noise
#                 elif noise_scheduler.config.prediction_type == "v_prediction":
#                     target = noise_scheduler.get_velocity(model_input, noise, timesteps)
#                 else:
#                     raise ValueError(
#                         f"Unknown prediction type {noise_scheduler.config.prediction_type}"
#                     )

#                 loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")


#                 # Gather the losses across all processes for logging (if we use distributed training).
#                 avg_loss = accelerator.gather(loss.repeat(train_batch_size)).mean()
#                 train_loss += avg_loss.item()

#                 # Backpropagate
#                 accelerator.backward(loss)
#                 if accelerator.sync_gradients:
#                     accelerator.clip_grad_norm_(params_to_optimize, 1)
#                 optimizer.step()
#                 lr_scheduler.step()
#                 optimizer.zero_grad()

#             # Checks if the accelerator has performed an optimization step behind the scenes
#             if accelerator.sync_gradients:
#                 progress_bar.update(1)
#                 global_step += 1
#                 accelerator.log({"train_loss": train_loss}, step=global_step)
#                 train_loss = 0.0

#                 if accelerator.is_main_process:
#                     if global_step % checkpointing_steps == 0:
#                         save_path = os.path.join(
#                             output_dir, f"checkpoint-{global_step}"
#                         )
#                         accelerator.save_state(save_path)
#                         logger.info(f"Saved state to {save_path}")

#             logs = {
#                 "step_loss": loss.detach().item(),
#                 "lr": lr_scheduler.get_last_lr()[0],
#             }
#             progress_bar.set_postfix(**logs)

#             if global_step >= max_train_steps:
#                 break
#     # Training 종료
#     # Save the lora layers
#     accelerator.wait_for_everyone()
#     if accelerator.is_main_process:
#         unet = unwrap_model(unet)
#         unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
#         StableDiffusionXLPipeline.save_lora_weights(
#             save_directory=output_dir,
#             unet_lora_layers=unet_lora_state_dict,
#         )
#         del unet
#         torch.cuda.empty_cache()

# import accelerate

# accelerate.notebook_launcher(training_function, args=(custom_dataset, unet, vae, text_encoder_one, text_encoder_two), num_processes=0)

# pipeline_repo_id = "stabilityai/stable-diffusion-xl-base-1.0"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# pipeline = StableDiffusionXLPipeline.from_pretrained(
#                                                 pretrained_model_name_or_path = pipeline_repo_id,
#                                                 torch_dtype = torch.float16,
#                                                 variant = "fp16",
#                                                 use_safetensors = True,
#                                                 ).to(device)

# pipeline.load_lora_weights("./sd_lora")

# # state_dict = torch.load("./sd_lora/pytorch_lora_weights.safetensors.bin")
# # pipeline.unet.load_state_dict(state_dict, strict=False)

# inference_prompt = "human statue"
# # inference_prompt = "16bit illustration of apple"
# images = pipeline(inference_prompt, num_images_per_prompt=3).images
# make_image_grid(images, rows=1, cols=3)
