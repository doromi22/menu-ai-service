import io
import modal
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
from fastapi import UploadFile, File, Form
from fastapi.responses import Response

def pre_download_models():
    import torch
    from diffusers import AutoPipelineForText2Image
    from huggingface_hub import hf_hub_download
    from rembg import new_session
    
    new_session("birefnet-general", providers=['CPUExecutionProvider'])
    
    AutoPipelineForText2Image.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    )
    hf_hub_download("ByteDance/SDXL-Lightning", "sdxl_lightning_4step_lora.safetensors")

image = (
    modal.Image.from_registry("nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04", add_python="3.11")
    .env({
        "TRANSFORMERS_NO_ADVISORY_WARNINGS": "1",
        "DISABLE_TELEMETRY": "1"
    })
    .pip_install(
        "torch==2.3.1",
        "torchvision==0.18.1",
        "diffusers==0.30.0",
        "transformers==4.44.0",
        "accelerate==0.33.0",
        "rembg==2.0.59",
        "onnxruntime-gpu==1.18.0",
        "fastapi==0.111.0",
        "python-multipart==0.0.9",
        "pillow==10.4.0",
        "peft",
        "huggingface-hub"
    )
    .run_function(pre_download_models)
)

app = modal.App("menu-ai-engine", image=image)

PROMPT_PRESETS = {
    "izakaya": (
        "modern minimalist japandi dining table, close-up completely bare smooth matte black wood surface filling the bottom frame, "
        "zero props, clean empty tabletop, soft ambient rim lighting, wabi-sabi aesthetic, highly detailed commercial product photography, neutral muted tones, appetizing delicious food, 8k"
    ),
    "cafe": (
        "modern minimalist cafe table, macro close-up completely bare smooth matte beige concrete surface taking up the entire table area, "
        "zero props, soft diffused window daylight, clean empty tabletop, subtle pastel neutral tones, kinfolk aesthetic, sleek and simple commercial product photography, appetizing delicious food, 8k"
    ),
    "ramen": (
        "ultra-minimalist dark dining table, close-up completely bare smooth matte charcoal slate surface, "
        "zero props, soft dramatic spotlight, clean empty tabletop, high-end fine dining aesthetic, sleek modern commercial product photography, appetizing delicious food, 8k"
    )
}

ANGLE_MODIFIERS = {
    "front": "straight-on horizontal eye-level shot, 85mm lens, smooth creamy blurred backdrop, flat surface plane in lower half",
    "45": "45 degree high angle perspective, 50mm f/2.8 lens, tabletop plane dominating the composition, clean surface depth",
    "top": "vertical flat lay, top-down 90 degree overhead shot, flat textured tabletop surface filling the whole screen completely"
}

COMMON_NEGATIVE = (
    "cutting board, chopping board, wooden board, tray, placemat, coaster, napkin, cloth, ingredients, props, crumbs, "
    "room, interior, windows, chairs, floor, building, ceiling, architecture, furniture, distant view, wide shot, "
    "people, hands, text, logo, watermark, cartoon, render, busy texture, messy, fake"
)

@app.cls(gpu="A10G", memory=16384, scaledown_window=300)
class FoodRenderer:
    @modal.enter()
    def setup(self):
        import torch
        from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image, EulerDiscreteScheduler
        from rembg import new_session

        self.rembg_session = new_session("birefnet-general", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        
        self.pipe_t2i = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        ).to("cuda")

        self.pipe_t2i.load_lora_weights("ByteDance/SDXL-Lightning", weight_name="sdxl_lightning_4step_lora.safetensors")
        self.pipe_t2i.fuse_lora()
        
        self.pipe_t2i.scheduler = EulerDiscreteScheduler.from_config(
            self.pipe_t2i.scheduler.config, 
            timestep_spacing="trailing"
        )

        self.pipe_i2i = AutoPipelineForImage2Image.from_pipe(self.pipe_t2i)

    @modal.fastapi_endpoint(method="POST")
    async def segment(
        self,
        image: UploadFile = File(...),
        prompt: str = Form("izakaya"),
        angle: str = Form("45")
    ):
        import torch
        from rembg import remove

        contents = await image.read()
        raw_image = Image.open(io.BytesIO(contents)).convert("RGBA")
        raw_image = ImageOps.contain(raw_image, (1024, 1024))
        w, h = (raw_image.width // 8) * 8, (raw_image.height // 8) * 8
        raw_image = raw_image.resize((w, h), Image.Resampling.LANCZOS)

        # 1. 누끼 추출
        rgba_dish = remove(raw_image, session=self.rembg_session)
        dish_rgb = rgba_dish.convert("RGB")
        dish_alpha = rgba_dish.split()[3]

        prompt_val = (prompt or "").lower()
        if "ramen" in prompt_val or "stone" in prompt_val:
            preset_key = "ramen"
        elif "cafe" in prompt_val or "bright" in prompt_val:
            preset_key = "cafe"
        else:
            preset_key = "izakaya"

        angle_key = angle if angle in ANGLE_MODIFIERS else "45"
        final_prompt = f"{PROMPT_PRESETS[preset_key]}, {ANGLE_MODIFIERS[angle_key]}"

        # 2. 음식 원본 화질 펌핑 (후반 덮어쓰기용)
        dish_enhanced = dish_rgb.copy()
        dish_enhanced = ImageEnhance.Color(dish_enhanced).enhance(1.2)
        dish_enhanced = ImageEnhance.Contrast(dish_enhanced).enhance(1.1)
        dish_enhanced = ImageEnhance.Sharpness(dish_enhanced).enhance(1.5) # 선명도를 높여 밥알 보존

        # 3. 깨끗한 배경 생성
        with torch.inference_mode():
            generated_bg = self.pipe_t2i(
                prompt=final_prompt,
                negative_prompt=COMMON_NEGATIVE,
                width=w,
                height=h,
                num_inference_steps=4,
                guidance_scale=0
            ).images[0].convert("RGBA")

        # 4. 물리적 콜라주 (그림자 포함)
        shadow_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        
        ambient_mask = dish_alpha.filter(ImageFilter.GaussianBlur(radius=15))
        ambient_offset = (0, 15) if angle_key != "top" else (0, 0)
        shadow_layer.paste(Image.new("RGBA", (w, h), (15, 15, 20, 120)), ambient_offset, ambient_mask)

        core_mask = dish_alpha.filter(ImageFilter.GaussianBlur(radius=4))
        core_offset = (0, 4) if angle_key != "top" else (0, 0)
        shadow_layer.paste(Image.new("RGBA", (w, h), (10, 10, 12, 180)), core_offset, core_mask)

        soft_dish_alpha = dish_alpha.filter(ImageFilter.GaussianBlur(radius=1.5))
        
        collage_canvas = Image.alpha_composite(generated_bg, shadow_layer)
        collage_canvas.paste(dish_enhanced, (0, 0), soft_dish_alpha)
        collage_rgb = collage_canvas.convert("RGB")

        # 5. 글로벌 하모나이제이션 (AI를 이용해 빛과 그림자 융합)
        with torch.inference_mode():
            harmonized_image = self.pipe_i2i(
                prompt=final_prompt,
                negative_prompt=COMMON_NEGATIVE,
                image=collage_rgb,
                num_inference_steps=4,
                strength=0.30, # 30% 개입 (외곽선 융합)
                guidance_scale=0
            ).images[0].convert("RGBA")

        # 6. [NEW] 코어 텍스처 프로텍션 (내부 화질 원상복구)
        # 마스크를 안쪽으로 11픽셀 정도 깎아내서 가장자리는 AI 합성을 유지하고 중심부만 보호합니다.
        shrunk_alpha = dish_alpha.filter(ImageFilter.MinFilter(11)) 
        core_soft_mask = shrunk_alpha.filter(ImageFilter.GaussianBlur(radius=5))
        
        # AI가 렌더링한 결과물 위에 중심부 밥알/질감을 원본으로 덮어버립니다.
        harmonized_image.paste(dish_enhanced.convert("RGBA"), (0, 0), core_soft_mask)

        buffer = io.BytesIO()
        harmonized_image.convert("RGB").save(buffer, format="JPEG", quality=95, optimize=True)
        return Response(content=buffer.getvalue(), media_type="image/jpeg")