"""PromptCraft Inference and Model Engine.

Handles local model loading, GPU/CPU execution, prompt generation,
tag extraction, and platform-specific formatting.
"""

from __future__ import annotations

import gc
import html
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel, GPT2TokenizerFast

BASE_DIR = Path(__file__).resolve().parent

# Candidate paths for the trained GPT-2 image prompt model
CANDIDATE_PATHS = [
    BASE_DIR / "model" / "trained_model" / "trained_model",
    BASE_DIR / "model" / "trained_model",
    BASE_DIR / "models" / "image_prompt_model",
    BASE_DIR / "model" / "image_prompt_model.pth",
    BASE_DIR / "models" / "image_prompt_model.pth",
    BASE_DIR / "model" / "image_prompt_model.pkl",
]

TOKENIZER_CANDIDATES = [
    BASE_DIR / "model" / "trained_model" / "trained_model",
    BASE_DIR / "model" / "tokenizer" / "tokenizer",
    BASE_DIR / "model" / "tokenizer",
    BASE_DIR / "models" / "tokenizer",
    "openai-community/gpt2",
]


class ModelLoadError(RuntimeError):
    """Raised when the trained model cannot be loaded."""


def get_device() -> Tuple[torch.device, str]:
    """Returns the compute device and friendly display name."""
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        return torch.device("cuda"), f"GPU: {name}"
    return torch.device("cpu"), "CPU"


def locate_model_source() -> Tuple[Path, str]:
    """Finds the first existing model file or directory."""
    for path in CANDIDATE_PATHS:
        if path.is_dir() and (path / "config.json").is_file():
            return path, "hf_directory"
        if path.is_file() and path.suffix.lower() in (".pth", ".pt"):
            return path, "pth_checkpoint"
        if path.is_file() and path.suffix.lower() == ".pkl":
            return path, "pkl_checkpoint"

    raise ModelLoadError(
        "Could not find any trained model files. Checked:\n"
        + "\n".join(f" - {p}" for p in CANDIDATE_PATHS)
    )


def locate_tokenizer_source() -> str | Path:
    """Finds the local tokenizer directory if present."""
    for candidate in TOKENIZER_CANDIDATES:
        if isinstance(candidate, Path) and candidate.is_dir():
            if (candidate / "tokenizer.json").is_file() or (candidate / "vocab.json").is_file():
                return candidate
    return "openai-community/gpt2"


def load_model(
    device_override: Optional[str] = None,
) -> Tuple[Any, Any, torch.device, Dict[str, Any]]:
    """Loads the tokenizer and fine-tuned GPT-2 model entirely locally."""
    model_path, model_type = locate_model_source()
    device, device_name = get_device()
    if device_override:
        device = torch.device(device_override)

    # 1. Load Tokenizer
    tok_source = locate_tokenizer_source()
    try:
        tokenizer = AutoTokenizer.from_pretrained(tok_source, local_files_only=isinstance(tok_source, Path))
    except Exception:
        try:
            tokenizer = GPT2TokenizerFast.from_pretrained(tok_source)
        except Exception as exc:
            raise ModelLoadError(f"Failed loading tokenizer from {tok_source}: {exc}") from exc

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load Model
    start_time = time.time()
    try:
        if model_type == "hf_directory":
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                local_files_only=True,
                dtype=torch.float32,
            )
        elif model_type == "pth_checkpoint":
            # Instantiate base architecture and load weights
            model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
            try:
                state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
            except (TypeError, Exception):
                state_dict = torch.load(model_path, map_location="cpu")

            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            model.load_state_dict(state_dict)
        elif model_type == "pkl_checkpoint":
            import pickle

            with open(model_path, "rb") as f:
                model = pickle.load(f)
        else:
            raise ModelLoadError(f"Unknown model type: {model_type}")

    except Exception as exc:
        raise ModelLoadError(f"Error loading model from {model_path}: {exc}") from exc

    model.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    model.to(device)
    model.eval()

    load_seconds = time.time() - start_time
    total_params = sum(p.numel() for p in model.parameters())

    meta = {
        "model_path": str(model_path),
        "model_type": model_type,
        "tokenizer_source": str(tok_source),
        "device": device_name,
        "parameters": total_params,
        "load_time_sec": round(load_seconds, 2),
        "vocab_size": getattr(model.config, "vocab_size", len(tokenizer)),
        "layers": getattr(model.config, "n_layer", 12),
        "embedding_dim": getattr(model.config, "n_embd", 768),
        "context_window": getattr(model.config, "n_positions", 1024),
    }

    return tokenizer, model, device, meta


def clean_prompt_text(text: str) -> str:
    """Cleans up formatting, duplicate commas, and erratic whitespace."""
    if not text:
        return ""
    # Normalize unicode spaces/quotes
    text = re.sub(r"[\xa0\u200b\u200e\u200f]", " ", text)
    # Fix repeated commas, colons, or periods
    text = re.sub(r",\s*,+", ", ", text)
    text = re.sub(r"\.\s*\.+", ". ", text)
    text = re.sub(r"\s+", " ", text).strip()
    # Remove trailing comma or period
    text = text.rstrip(", ")
    return text


TAG_PATTERNS = {
    "Style": [
        "photorealistic", "photo", "cinematic", "anime", "manga", "concept art",
        "digital art", "oil painting", "watercolor", "illustration", "cyberpunk",
        "steampunk", "dark fantasy", "fantasy", "surrealism", "render", "3d render",
        "minimalist", "retro", "comic book", "isometric"
    ],
    "Engine & 3D": [
        "octane render", "unreal engine", "unreal engine 5", "cgsociety",
        "blender", "vray", "houdini", "ray tracing", "global illumination"
    ],
    "Lighting": [
        "volumetric lighting", "god rays", "neon", "golden hour", "studio lighting",
        "bioluminescent", "soft lighting", "rim lighting", "dramatic lighting",
        "ambient light", "cinematic lighting", "chiaroscuro", "backlit"
    ],
    "Resolution & Quality": [
        "8k", "4k", "hyperdetailed", "ultra-detailed", "highly detailed",
        "masterpiece", "award winning", "contest winner", "sharp focus",
        "trending on artstation", "trending on pixiv"
    ],
    "Camera & Lens": [
        "wide angle", "macro", "close up", "telephoto", "bokeh", "depth of field",
        "low angle", "aerial view", "drone photography", "portrait", "85mm"
    ],
    "Artists": [
        "greg rutkowski", "wlop", "artgerm", "alphonse mucha", "makoto shinkai",
        "ilya kuvshinov", "ross tran", "james gurney", "okuda genso", "moebius"
    ]
}


def extract_prompt_tags(prompt_text: str) -> Dict[str, List[str]]:
    """Analyzes a generated prompt and categorizes detected descriptive tags."""
    lower_text = prompt_text.lower()
    detected: Dict[str, List[str]] = {}

    for category, keywords in TAG_PATTERNS.items():
        found = []
        for kw in keywords:
            if kw in lower_text:
                found.append(kw.title())
        if found:
            detected[category] = found

    return detected


def get_negative_prompt(style: str) -> str:
    """Returns recommended negative prompt keywords based on style."""
    base = "low quality, bad quality, blurry, distorted, watermark, signature, ugly, bad anatomy, deformed"
    s = style.lower()
    if "photo" in s or "realistic" in s or "portrait" in s:
        return f"{base}, cartoon, anime, 3d render, extra limbs, extra fingers, poor lighting, oversaturated, oversmoothed"
    elif "anime" in s or "manga" in s:
        return f"{base}, 3d render, realistic photo, bad lineart, poor coloring, duplicate faces"
    elif "3d" in s or "octane" in s:
        return f"{base}, 2d sketch, flat colors, low polygon, noisy, low resolution"
    return f"{base}, extra fingers, missing limbs, cropped, out of frame, grainy"


def format_for_platform(
    prompt: str,
    platform: str,
    aspect_ratio: str = "16:9",
    stylize_val: int = 100,
) -> str:
    """Formats the prompt specifically for the target image generation tool."""
    clean = clean_prompt_text(prompt)
    plat = platform.lower()

    if "midjourney" in plat:
        ar_flag = f"--ar {aspect_ratio}" if aspect_ratio else "--ar 16:9"
        v_flag = "--v 6.0"
        s_flag = f"--s {stylize_val}" if stylize_val != 100 else ""
        flags = " ".join(f for f in [ar_flag, v_flag, s_flag] if f)
        return f"/imagine prompt: {clean} {flags}"

    elif "stable diffusion" in plat or "sdxl" in plat:
        return f"{clean}, (masterpiece:1.2), (high quality:1.2), highly detailed"

    elif "flux" in plat:
        # Flux performs best with fluent, punctuation-separated natural phrases
        return clean

    elif "dall-e" in plat or "dalle" in plat:
        return f"A high-quality image of {clean}. Vivid colors, meticulous detail, and artistic atmosphere."

    return clean


def generate_image_prompts(
    seed: str,
    tokenizer: Any,
    model: Any,
    device: torch.device,
    *,
    style: str = "None",
    lighting: str = "None",
    camera: str = "None",
    boosters: Optional[List[str]] = None,
    target_platform: str = "Standard",
    aspect_ratio: str = "16:9",
    num_variations: int = 1,
    max_new_tokens: int = 60,
    temperature: float = 0.85,
    top_p: float = 0.92,
    top_k: int = 50,
    repetition_penalty: float = 1.15,
    seed_int: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Generates one or more prompt continuations from the local fine-tuned model."""
    cleaned_seed = seed.strip()
    if not cleaned_seed:
        raise ValueError("Please provide a seed concept or subject.")

    # Build the initial prompt conditioning prefix
    parts = [cleaned_seed]
    if style and style != "None":
        parts.append(style)
    if lighting and lighting != "None":
        parts.append(lighting)
    if camera and camera != "None":
        parts.append(camera)
    if boosters:
        parts.extend(boosters)

    model_input = ", ".join(p.strip() for p in parts if p.strip())

    if seed_int is not None:
        torch.manual_seed(seed_int)

    context_limit = getattr(model.config, "n_positions", 1024) or 1024
    encoded = tokenizer(
        model_input,
        return_tensors="pt",
        truncation=True,
        max_length=max(1, context_limit - 1),
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    input_len = encoded["input_ids"].shape[-1]
    allowed_tokens = min(int(max_new_tokens), max(1, context_limit - input_len))

    gen_kwargs = {
        "max_new_tokens": allowed_tokens,
        "do_sample": True,
        "temperature": max(float(temperature), 0.05),
        "top_p": min(max(float(top_p), 0.05), 1.0),
        "repetition_penalty": float(repetition_penalty),
        "no_repeat_ngram_size": 3,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
        "num_return_sequences": max(1, min(int(num_variations), 4)),
    }
    if top_k and int(top_k) > 0:
        gen_kwargs["top_k"] = int(top_k)

    with torch.inference_mode():
        output_ids = model.generate(**encoded, **gen_kwargs)

    results = []
    for idx, seq in enumerate(output_ids):
        # Decode only the generated portion
        continuation = tokenizer.decode(seq[input_len:], skip_special_tokens=True).strip()
        full_raw = tokenizer.decode(seq, skip_special_tokens=True).strip()

        # Build clean merged prompt
        if continuation:
            raw_combined = f"{model_input}, {continuation}"
        else:
            raw_combined = full_raw

        cleaned = clean_prompt_text(raw_combined)
        platform_formatted = format_for_platform(cleaned, target_platform, aspect_ratio=aspect_ratio)
        tags = extract_prompt_tags(cleaned)
        neg_prompt = get_negative_prompt(style)

        results.append({
            "variation_index": idx + 1,
            "prompt": cleaned,
            "continuation": clean_prompt_text(continuation),
            "formatted_for_platform": platform_formatted,
            "target_platform": target_platform,
            "negative_prompt": neg_prompt,
            "tags": tags,
            "tokens_generated": len(seq) - input_len,
        })

    return results


def build_llm_prompt(
    idea: str,
    category: str,
    audience: str,
    tone: str,
    output_format: str,
    detail_level: str,
    constraints: str = "",
) -> str:
    """Builds a structured prompt template for LLMs (ChatGPT, Claude, Gemini)."""
    constraints_text = constraints.strip() or "Standard quality guidelines apply."

    return f"""## Role & Context
You are a top-tier {category.lower()} specialist and expert prompt engineer.
Your goal is to provide a complete, robust, and well-thought-out solution.

## User Objective
{idea.strip()}

## Target Audience
{audience}

## Communication Style & Tone
Adopt a {tone.lower()} tone. Prioritize clarity, precision, and practical utility.

## Expected Response Structure
- Format: {output_format}
- Depth of Explanation: {detail_level}

## Guidelines & Constraints
1. Carefully address every part of the objective before synthesizing the answer.
2. Provide step-by-step reasoning where applicable.
3. If writing code or technical content, produce complete, runnable, and documented examples.
4. Highlight key caveats, edge cases, and best practices.
5. Specific Requirements: {constraints_text}

Begin your response now:"""
