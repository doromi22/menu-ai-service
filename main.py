import io
import os
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import Response
from rembg import remove, new_session
from PIL import Image, ImageOps, ImageFilter
from optimum.onnxruntime import ORTStableDiffusionInpaintPipeline

app = FastAPI(title="MenuAI ONNX Runtime Engine")

# 1. ONNX 가속 정밀 세그멘테이션 세션
# rembg 내부가 onnxruntime-gpu로 최적화되어 구동됩니다.
rembg_session = new_session("birefnet-general")

# 2. ONNX Inpainting 파이프라인 로드 (경량 ONNX 추론 전용)
# 첫 실행 시 ONNX 변환 가중치를 로드합니다.
pipe = ORTStableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    export=False,
    provider="CUDAExecutionProvider"  # DirectML 사용 시 DmlExecutionProvider
)

COMMON_NEGATIVE = (
    "extra bowls, extra plates, duplicate dishes, additional food items, chopsticks, forks, spoons, "
    "cups, drinks, napkins, people, hands, text, watermark, logo, cartoon, render, blurry, messy, low quality"
)

PROMPT_PRESETS = {
    "izakaya": (
        "placed on a clean empty rustic dark oak wood table surface, warm subtle ambient lighting, cozy izakaya atmosphere, natural soft ground shadow under the bowl, authentic texture, photorealistic studio shot",
        COMMON_NEGATIVE
    ),
    "cafe": (
        "placed on a clean empty modern white marble table surface, soft diffused morning sunlight, bright aesthetic cafe tabletop, subtle contact shadow, elegant high-end photography",
        COMMON_NEGATIVE
    ),
    "ramen": (
        "placed on a clean empty matte black slate stone counter table, dramatic moody lighting, authentic ramen bar counter texture, soft realistic ambient occlusion shadow",
        COMMON_NEGATIVE
    )
}

@app.post("/segment")
async def generate_food_scene(
    image: UploadFile = File(...),
    prompt: str = Form("izakaya")
):
    contents = await image.read()
    init_image = Image.open(io.BytesIO(contents)).convert("RGB")

    # 1. 512x512 또는 768x768 표준 해상도 리사이즈 (ONNX 인페인팅 최적 비율)
    init_image = ImageOps.contain(init_image, (768, 768))
    w, h = (init_image.width // 8) * 8, (init_image.height // 8) * 8
    init_image = init_image.resize((w, h), Image.Resampling.LANCZOS)

    # 2. BiRefNet 누끼 추출
    rgba = remove(init_image, session=rembg_session)
    alpha = rgba.split()[3]

    # 3. 마스크 생성 및 미세 경계 확장(Dilation)
    dilated_alpha = alpha.filter(ImageFilter.MaxFilter(size=5))
    mask_image = ImageOps.invert(dilated_alpha)

    # 4. 프롬프트 매핑
    preset_key = "ramen" if "ramen" in prompt.lower() or "stone" in prompt.lower() else (
        "cafe" if "cafe" in prompt.lower() or "bright" in prompt.lower() else "izakaya"
    )
    pos_prompt, neg_prompt = PROMPT_PRESETS[preset_key]

    # 5. ONNX 고속 추론
    generated_scene = pipe(
        prompt=pos_prompt,
        negative_prompt=neg_prompt,
        image=init_image,
        mask_image=mask_image,
        num_inference_steps=20,
        guidance_scale=7.5
    ).images[0]

    # 6. 음식 본체 원본 100% 덮어쓰기 (경품표시법 준수)
    final_output = generated_scene.copy()
    final_output.paste(init_image, (0, 0), alpha)

    buffer = io.BytesIO()
    final_output.save(buffer, format="JPEG", quality=95, optimize=True)
    return Response(content=buffer.getvalue(), media_type="image/jpeg")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)