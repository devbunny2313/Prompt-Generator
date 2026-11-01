"""PromptCraft AI Studio - Modern UI for fine-tuned GPT-2 Image Prompt Model.

Features:
- Image Prompt Studio with fine-tuned GPT-2 DiffusionDB checkpoint
- Interactive style, lighting, camera, and artist modifiers
- Multi-variation generation (1-4 variations)
- Platform decorators (Midjourney v6, SDXL, Flux.1, DALL-E 3)
- Automatic negative prompt generator and prompt tag breakdown
- A/B Model Comparison (side-by-side creativity & sampling experiments)
- LLM Prompt Architect (structured prompt engineering)
- Session Prompt Library with search, filter, and JSON/TXT export
- Model Diagnostics & live speed benchmarking
"""

from __future__ import annotations

from datetime import datetime
import html
import json
from pathlib import Path
import time

import streamlit as st

import engine

# ============================================================
# APP CONFIGURATION & THEMING
# ============================================================

st.set_page_config(
    page_title="PromptCraft | AI Prompt Studio",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished, modern dark theme styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&family=Space+Grotesk:wght@600;700&display=swap');

    :root {
        --bg-main: #0a0d14;
        --bg-panel: #111724;
        --bg-panel-2: #162032;
        --border-color: rgba(148, 163, 184, 0.14);
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
        --accent-purple: #8b5cf6;
        --accent-cyan: #38bdf8;
        --accent-emerald: #10b981;
    }

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    .stApp {
        background:
            radial-gradient(circle at 10% 0%, rgba(139, 92, 246, 0.12) 0%, transparent 40%),
            radial-gradient(circle at 90% 10%, rgba(56, 189, 248, 0.08) 0%, transparent 35%),
            #0a0d14;
        color: var(--text-primary);
    }

    /* Top header area */
    [data-testid="stHeader"] {
        background: transparent;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #0e131f !important;
        border-right: 1px solid var(--border-color);
    }

    .sidebar-brand {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.35rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        color: #ffffff;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .sidebar-brand span {
        background: linear-gradient(135deg, #a78bfa 0%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Status badge pill */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        background: rgba(16, 185, 129, 0.12);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.28);
        margin-top: 4px;
        margin-bottom: 12px;
    }
    .status-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background-color: #34d399;
        box-shadow: 0 0 8px #34d399;
    }

    /* Hero Banner */
    .hero-box {
        padding: 1.8rem 2.2rem;
        border: 1px solid rgba(148, 163, 184, 0.16);
        border-radius: 20px;
        background: linear-gradient(135deg, rgba(22, 32, 50, 0.85) 0%, rgba(14, 20, 34, 0.95) 100%);
        box-shadow: 0 12px 36px rgba(0, 0, 0, 0.35);
        margin-bottom: 1.5rem;
    }
    .hero-eyebrow {
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: #a78bfa;
        margin-bottom: 0.4rem;
    }
    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        color: #ffffff;
        margin: 0;
        line-height: 1.2;
    }
    .hero-desc {
        color: var(--text-secondary);
        font-size: 0.92rem;
        margin-top: 0.5rem;
        margin-bottom: 0;
    }

    /* Spec cards */
    .stat-card {
        background: rgba(18, 25, 38, 0.7);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.75rem;
    }
    .stat-label {
        color: var(--text-secondary);
        font-size: 0.74rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .stat-value {
        color: #f1f5f9;
        font-size: 0.96rem;
        font-weight: 700;
        margin-top: 0.2rem;
    }

    /* Tag pills */
    .tag-pill {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.72rem;
        font-weight: 600;
        margin: 2px 4px 2px 0;
    }
    .tag-style { background: rgba(59, 130, 246, 0.18); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.3); }
    .tag-engine { background: rgba(16, 185, 129, 0.18); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }
    .tag-quality { background: rgba(245, 158, 11, 0.18); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }
    .tag-camera { background: rgba(236, 72, 153, 0.18); color: #f472b6; border: 1px solid rgba(236, 72, 153, 0.3); }
    .tag-artist { background: rgba(168, 85, 247, 0.18); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.3); }

    /* Primary buttons */
    .stButton > button {
        border-radius: 10px;
        font-weight: 700;
        font-size: 0.95rem;
        min-height: 2.75rem;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(139, 92, 246, 0.25);
    }

    /* Quick idea pills */
    div[data-testid="stHorizontalBlock"] button[kind="secondary"] {
        font-size: 0.8rem;
        padding: 0.2rem 0.6rem;
        min-height: 2rem;
        border-radius: 20px;
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(148, 163, 184, 0.2);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MODEL CACHING & SESSION STATE
# ============================================================

@st.cache_resource(show_spinner="Initializing fine-tuned GPT-2 checkpoint...")
def get_cached_model():
    """Loads and caches the model & tokenizer once for fast inference."""
    return engine.load_model()


# Initialize session state variables
if "history" not in st.session_state:
    st.session_state.history = []

if "current_image_results" not in st.session_state:
    st.session_state.current_image_results = []

if "text_llm_result" not in st.session_state:
    st.session_state.text_llm_result = ""

if "ab_result_a" not in st.session_state:
    st.session_state.ab_result_a = ""

if "ab_result_b" not in st.session_state:
    st.session_state.ab_result_b = ""

if "active_seed" not in st.session_state:
    st.session_state.active_seed = ""


def add_to_history(kind: str, prompt: str, seed: str = "", metadata: dict | None = None):
    """Saves a prompt to the session history library."""
    entry = {
        "id": len(st.session_state.history) + 1,
        "type": kind,
        "seed": seed,
        "prompt": prompt,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "metadata": metadata or {},
    }
    st.session_state.history.insert(0, entry)


# Pre-load or check model status
try:
    tokenizer, model, device, model_meta = get_cached_model()
    model_ready = True
    model_error_msg = ""
except Exception as exc:
    model_ready = False
    tokenizer, model, device = None, None, None
    model_meta = {}
    model_error_msg = str(exc)


# ============================================================
# SIDEBAR NAVIGATION & SYSTEM TELEMETRY
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            ✨ Prompt<span>Craft</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if model_ready:
        st.markdown(
            f"""
            <div class="status-badge">
                <span class="status-dot"></span>
                <span>GPT-2 Checkpoint Online</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.error("Model Offline")

    # Main Navigation
    page = st.radio(
        "STUDIO NAVIGATION",
        [
            "🎨 Image Prompt Studio",
            "⚖️ A/B Comparison",
            "✍️ LLM Prompt Architect",
            "📚 Prompt Library",
            "🔬 Model Diagnostics",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    # Hardware & Model Specs Card
    st.markdown(
        """
        <div class="stat-card">
            <div class="stat-label">Hardware Device</div>
            <div class="stat-value">{}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Model Parameters</div>
            <div class="stat-value">{}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Format / Architecture</div>
            <div class="stat-value">Safetensors (GPT-2)</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Prompts in Session</div>
            <div class="stat-value">{} items</div>
        </div>
        """.format(
            html.escape(model_meta.get("device", "CPU")),
            f"{model_meta.get('parameters', 124439808):,} (~124M)",
            len(st.session_state.history),
        ),
        unsafe_allow_html=True,
    )

    if st.button("🔄 Reload Model Weights", use_container_width=True):
        get_cached_model.clear()
        st.session_state.current_image_results = []
        st.rerun()


# ============================================================
# 1. IMAGE PROMPT STUDIO (CORE MODEL UI)
# ============================================================

if page == "🎨 Image Prompt Studio":
    st.markdown(
        """
        <div class="hero-box">
            <div class="hero-eyebrow">FINE-TUNED DIFFUSIONDB MODEL</div>
            <h1 class="hero-title">Image Prompt Studio 🎨</h1>
            <p class="hero-desc">Expand short creative concepts into rich, atmospheric art prompts using your fine-tuned GPT-2 checkpoint.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not model_ready:
        st.error(f"Cannot load trained model: {model_error_msg}")
        st.info("Ensure checkpoint files are present in `model/trained_model/trained_model/` or `model/`.")
        st.stop()

    # Two main columns: Controls (left) and Results (right)
    col_input, col_output = st.columns([1.1, 1.2], gap="large")

    with col_input:
        st.subheader("1. Concept & Subject")

        # Quick inspiration seeds
        st.caption("Quick Inspiration Starters (click to fill):")
        chip_col1, chip_col2, chip_col3 = st.columns(3)
        with chip_col1:
            if st.button("🏙️ Cyberpunk Samurai", key="chip1", use_container_width=True):
                st.session_state.active_seed = "a futuristic cyberpunk samurai standing on a neon rooftop in rainy Neo-Tokyo"
                st.rerun()
            if st.button("🏰 Floating Castle", key="chip4", use_container_width=True):
                st.session_state.active_seed = "an ancient ethereal castle perched atop floating islands amidst clouds"
                st.rerun()
        with chip_col2:
            if st.button("🌊 Cosmic Jellyfish", key="chip2", use_container_width=True):
                st.session_state.active_seed = "a giant bioluminescent jellyfish gliding through deep space nebula"
                st.rerun()
            if st.button("🤖 Android Portrait", key="chip5", use_container_width=True):
                st.session_state.active_seed = "a hyperrealistic portrait of a porcelain android with visible gold circuits"
                st.rerun()
        with chip_col3:
            if st.button("🌲 Steampunk Forest", key="chip3", use_container_width=True):
                st.session_state.active_seed = "a mechanical owl resting inside a foggy enchanted steampunk forest"
                st.rerun()
            if st.button("🐉 Crystal Dragon", key="chip6", use_container_width=True):
                st.session_state.active_seed = "a majestic translucent crystal dragon resting atop a cavern of glowing gems"
                st.rerun()

        seed_text = st.text_area(
            "Enter your seed concept or subject:",
            value=st.session_state.active_seed,
            placeholder="e.g. A solitary astronaut discovering glowing alien ruins on Mars...",
            height=110,
            help="Your base idea. The model was trained on DiffusionDB prompts to enrich seed descriptions into full image prompts.",
        )
        st.session_state.active_seed = seed_text

        st.subheader("2. Creative Modifiers")
        mod_col1, mod_col2 = st.columns(2)
        with mod_col1:
            style_opt = st.selectbox(
                "Artistic Style",
                [
                    "None",
                    "Cinematic",
                    "Photorealistic",
                    "Anime / Manga",
                    "3D Octane Render",
                    "Fantasy Concept Art",
                    "Dark Fantasy",
                    "Cyberpunk",
                    "Watercolor",
                    "Surrealism",
                    "Retro Sci-Fi",
                    "Oil Painting",
                    "Minimalist",
                    "Vintage Polaroid",
                ],
                index=1,
            )
            lighting_opt = st.selectbox(
                "Lighting & Atmosphere",
                [
                    "None",
                    "Volumetric Lighting",
                    "God Rays & Golden Hour",
                    "Cyberpunk Neon Glow",
                    "Bioluminescent Light",
                    "Moody Cinematic Shadows",
                    "Ethereal Dawn Mist",
                    "Studio Softbox & Rim Light",
                    "Dramatic Chiaroscuro",
                ],
                index=1,
            )

        with mod_col2:
            camera_opt = st.selectbox(
                "Camera & Framing",
                [
                    "None",
                    "Wide Angle Landscape",
                    "Macro Close-up",
                    "Low Angle Hero Perspective",
                    "85mm Portrait Bokeh",
                    "Drone Aerial View",
                    "Cinematic Dutch Angle",
                    "Eye-Level Symmetrical",
                ],
                index=0,
            )
            target_platform = st.selectbox(
                "Target AI Generator",
                [
                    "Standard (Raw DiffusionDB)",
                    "Midjourney v6",
                    "Stable Diffusion / SDXL",
                    "Flux.1",
                    "DALL-E 3",
                ],
                index=1,
                help="Automatically structures flags, tags, and formatting for the chosen image generation platform.",
            )

        aspect_ratio = "16:9"
        if "Midjourney" in target_platform or "SDXL" in target_platform:
            aspect_ratio = st.select_slider(
                "Aspect Ratio",
                options=["16:9", "21:9", "1:1", "4:5", "9:16"],
                value="16:9",
            )

        boosters = st.multiselect(
            "Quality & Engine Boosters",
            [
                "Trending on ArtStation",
                "Octane Render",
                "Unreal Engine 5",
                "CGSociety Contest Winner",
                "Masterpiece",
                "8k Resolution",
                "Highly Detailed",
                "Award Winning",
                "WLOP & Greg Rutkowski Style",
            ],
            default=["Trending on ArtStation", "Octane Render"],
        )

        with st.expander("⚙️ Advanced Model Controls"):
            adv_c1, adv_c2 = st.columns(2)
            with adv_c1:
                temperature = st.slider(
                    "Creativity (Temperature)",
                    min_value=0.2,
                    max_value=1.5,
                    value=0.85,
                    step=0.05,
                    help="Lower values are focused and conservative; higher values are varied and imaginative.",
                )
                top_p = st.slider(
                    "Nucleus Sampling (Top-p)",
                    min_value=0.5,
                    max_value=1.0,
                    value=0.92,
                    step=0.02,
                )
                top_k = st.slider("Top-k Filtering", min_value=0, max_value=100, value=50, step=5)

            with adv_c2:
                max_new_tokens = st.slider(
                    "Continuation Length (Tokens)",
                    min_value=20,
                    max_value=120,
                    value=60,
                    step=5,
                )
                repetition_penalty = st.slider(
                    "Repetition Penalty",
                    min_value=1.0,
                    max_value=1.5,
                    value=1.15,
                    step=0.05,
                )
                num_variations = st.slider("Number of Variations", min_value=1, max_value=4, value=2)

            use_fixed_seed = st.checkbox("Set Fixed Seed for Reproducibility")
            fixed_seed = None
            if use_fixed_seed:
                fixed_seed = st.number_input("Seed Value", min_value=0, max_value=9999999, value=42)

        # Generate Button
        generate_clicked = st.button(
            "✨ Generate Prompts with Model",
            type="primary",
            use_container_width=True,
        )

        if generate_clicked:
            if not seed_text.strip():
                st.warning("Please enter a concept or click one of the quick starters above.")
            else:
                try:
                    with st.spinner("Generating prompt continuations using fine-tuned GPT-2..."):
                        t_start = time.time()
                        variations = engine.generate_image_prompts(
                            seed=seed_text,
                            tokenizer=tokenizer,
                            model=model,
                            device=device,
                            style=style_opt,
                            lighting=lighting_opt,
                            camera=camera_opt,
                            boosters=boosters,
                            target_platform=target_platform,
                            aspect_ratio=aspect_ratio,
                            num_variations=num_variations,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            top_k=top_k,
                            repetition_penalty=repetition_penalty,
                            seed_int=fixed_seed,
                        )
                        elapsed = time.time() - t_start

                        st.session_state.current_image_results = variations
                        # Save first variation to history
                        for v in variations:
                            add_to_history(
                                kind="Image Prompt",
                                prompt=v["formatted_for_platform"],
                                seed=seed_text,
                                metadata={
                                    "style": style_opt,
                                    "platform": target_platform,
                                    "elapsed_sec": round(elapsed, 2),
                                },
                            )
                        st.success(f"Generated {len(variations)} variation(s) in {elapsed:.2f}s!")

                except Exception as exc:
                    st.error(f"Generation error: {exc}")

    with col_output:
        st.subheader("3. Model Results & Variations")

        if not st.session_state.current_image_results:
            st.info(
                "👈 Choose your concept, styles, and click **Generate Prompts with Model** to see results here.",
                icon="ℹ️",
            )
        else:
            for idx, res in enumerate(st.session_state.current_image_results):
                with st.container(border=True):
                    head_l, head_r = st.columns([1, 1])
                    with head_l:
                        st.markdown(f"**Variation #{idx + 1}** · {res['target_platform']}")
                    with head_r:
                        st.caption(f"{res.get('tokens_generated', 0)} tokens generated")

                    # Primary formatted output with native Streamlit copy button
                    st.code(res["formatted_for_platform"], language=None)

                    # Prompt Tag breakdown
                    tags = res.get("tags", {})
                    if tags:
                        pill_html = ""
                        for cat, words in tags.items():
                            badge_cls = "tag-style"
                            if cat == "Engine & 3D":
                                badge_cls = "tag-engine"
                            elif cat == "Resolution & Quality":
                                badge_cls = "tag-quality"
                            elif cat == "Camera & Lens":
                                badge_cls = "tag-camera"
                            elif cat == "Artists":
                                badge_cls = "tag-artist"

                            for w in words:
                                pill_html += f'<span class="tag-pill {badge_cls}">{html.escape(w)}</span>'

                        st.markdown(
                            f'<div style="margin-bottom:8px;">{pill_html}</div>',
                            unsafe_allow_html=True,
                        )

                    # Negative prompt for SDXL/SD
                    if res.get("negative_prompt"):
                        with st.expander("Recommended Negative Prompt"):
                            st.code(res["negative_prompt"], language=None)

                    # Download button
                    st.download_button(
                        f"⬇️ Download Variation #{idx + 1}",
                        data=res["formatted_for_platform"],
                        file_name=f"prompt_variation_{idx + 1}.txt",
                        mime="text/plain",
                        key=f"dl_var_{idx}_{time.time()}",
                        use_container_width=True,
                    )


# ============================================================
# 2. A/B MODEL COMPARISON
# ============================================================

elif page == "⚖️ A/B Comparison":
    st.markdown(
        """
        <div class="hero-box">
            <div class="hero-eyebrow">A/B EXPERIMENTATION</div>
            <h1 class="hero-title">A/B Model Comparison ⚖️</h1>
            <p class="hero-desc">Compare two different temperatures, styles, or sampling settings side-by-side on the exact same seed concept.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not model_ready:
        st.error(f"Cannot load trained model: {model_error_msg}")
        st.stop()

    ab_seed = st.text_input(
        "Shared Seed Concept:",
        value="a glowing cybernetic tiger in a futuristic jungle, octane render",
    )

    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.markdown("### Configuration A (e.g. Precise)")
        temp_a = st.slider("Temperature A", 0.2, 1.4, 0.6, 0.05, key="temp_a")
        style_a = st.selectbox(
            "Style A",
            ["Photorealistic", "Cinematic", "3D Octane Render", "Anime / Manga", "Dark Fantasy"],
            index=0,
            key="style_a",
        )
        topp_a = st.slider("Top-p A", 0.5, 1.0, 0.85, 0.05, key="topp_a")

    with col_b:
        st.markdown("### Configuration B (e.g. Creative)")
        temp_b = st.slider("Temperature B", 0.2, 1.4, 1.05, 0.05, key="temp_b")
        style_b = st.selectbox(
            "Style B",
            ["Photorealistic", "Cinematic", "3D Octane Render", "Anime / Manga", "Dark Fantasy"],
            index=1,
            key="style_b",
        )
        topp_b = st.slider("Top-p B", 0.5, 1.0, 0.95, 0.05, key="topp_b")

    if st.button("⚖️ Run Side-by-Side Comparison", type="primary", use_container_width=True):
        if not ab_seed.strip():
            st.warning("Enter a seed concept above.")
        else:
            with st.spinner("Generating both outputs simultaneously..."):
                res_a = engine.generate_image_prompts(
                    seed=ab_seed,
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    style=style_a,
                    temperature=temp_a,
                    top_p=topp_a,
                    num_variations=1,
                    max_new_tokens=60,
                )
                res_b = engine.generate_image_prompts(
                    seed=ab_seed,
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    style=style_b,
                    temperature=temp_b,
                    top_p=topp_b,
                    num_variations=1,
                    max_new_tokens=60,
                )
                st.session_state.ab_result_a = res_a[0]["prompt"]
                st.session_state.ab_result_b = res_b[0]["prompt"]

    if st.session_state.ab_result_a and st.session_state.ab_result_b:
        res_col_a, res_col_b = st.columns(2, gap="large")
        with res_col_a:
            with st.container(border=True):
                st.markdown(f"**Result A (Temp: {temp_a}, Style: {style_a})**")
                st.code(st.session_state.ab_result_a, language=None)
        with res_col_b:
            with st.container(border=True):
                st.markdown(f"**Result B (Temp: {temp_b}, Style: {style_b})**")
                st.code(st.session_state.ab_result_b, language=None)


# ============================================================
# 3. LLM PROMPT ARCHITECT
# ============================================================

elif page == "✍️ LLM Prompt Architect":
    st.markdown(
        """
        <div class="hero-box">
            <div class="hero-eyebrow">LANGUAGE MODEL ENGINEERING</div>
            <h1 class="hero-title">LLM Prompt Architect ✍️</h1>
            <p class="hero-desc">Structure high-performance prompts for ChatGPT, Claude, and Gemini with system roles, tasks, context, and clear constraints.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        task_idea = st.text_area(
            "What specific task or objective should the AI accomplish?",
            placeholder="e.g. Build an authentication system in FastAPI with JWT tokens, refresh tokens, and PostgreSQL...",
            height=110,
        )

        c1, c2 = st.columns(2)
        with c1:
            category = st.selectbox(
                "Category / Role Specialist",
                [
                    "Software Engineering",
                    "Data Science & Machine Learning",
                    "Technical Writing",
                    "Academic Research",
                    "Product Management",
                    "Cybersecurity & Auditing",
                    "Creative Writing",
                ],
            )
            audience = st.selectbox(
                "Target Audience",
                ["Senior Engineers", "Junior Developers", "Students / Beginners", "Business Stakeholders", "General Audience"],
            )
            tone = st.selectbox(
                "Tone & Style",
                ["Rigorous & Technical", "Clear & Pragmatic", "Friendly & Instructive", "Academic & Formal", "Concise & Direct"],
            )

        with c2:
            output_format = st.selectbox(
                "Expected Response Format",
                [
                    "Production Code with In-line Explanations",
                    "Step-by-Step Implementation Guide",
                    "Architectural Blueprint & Markdown Table",
                    "Comprehensive Technical Report",
                    "Checklist & Action Plan",
                ],
            )
            detail_level = st.select_slider(
                "Depth of Explanation",
                options=["Concise & Code-First", "Balanced", "Deep-Dive & Comprehensive"],
                value="Balanced",
            )
            constraints = st.text_area(
                "Specific Constraints & Edge Cases",
                placeholder="e.g. Must use Python 3.12 type hints, handle expired tokens, avoid third-party wrappers...",
                height=90,
            )

        if st.button("✨ Architect Structured Prompt", type="primary", use_container_width=True):
            if not task_idea.strip():
                st.warning("Please describe the task objective first.")
            else:
                prompt_result = engine.build_llm_prompt(
                    idea=task_idea,
                    category=category,
                    audience=audience,
                    tone=tone,
                    output_format=output_format,
                    detail_level=detail_level,
                    constraints=constraints,
                )
                st.session_state.text_llm_result = prompt_result
                add_to_history(
                    kind="LLM Prompt",
                    prompt=prompt_result,
                    seed=task_idea,
                    metadata={"category": category, "format": output_format},
                )
                st.success("Structured LLM prompt generated successfully!")

    if st.session_state.text_llm_result:
        st.subheader("Architected Prompt")
        st.code(st.session_state.text_llm_result, language="markdown")
        st.download_button(
            "⬇️ Download Prompt as Markdown",
            data=st.session_state.text_llm_result,
            file_name="structured_llm_prompt.md",
            mime="text/markdown",
            use_container_width=True,
        )


# ============================================================
# 4. PROMPT LIBRARY & SESSION HISTORY
# ============================================================

elif page == "📚 Prompt Library":
    st.markdown(
        """
        <div class="hero-box">
            <div class="hero-eyebrow">YOUR CREATIVE HISTORY</div>
            <h1 class="hero-title">Prompt Library 📚</h1>
            <p class="hero-desc">All prompts generated during your active session. Search, filter, inspect, and export your prompt collection.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    history = st.session_state.history
    if not history:
        st.info("No prompts generated in this session yet. Generate image or LLM prompts to see them cataloged here.")
    else:
        top_c1, top_c2, top_c3 = st.columns([1.5, 1, 1])
        with top_c1:
            search_query = st.text_input("🔍 Search prompts...", placeholder="Search keywords...")
        with top_c2:
            filter_type = st.selectbox("Filter Type", ["All Types", "Image Prompt", "LLM Prompt"])
        with top_c3:
            st.metric("Total Saved", f"{len(history)} prompts")

        # Filter prompts
        filtered = []
        for item in history:
            if filter_type != "All Types" and item["type"] != filter_type:
                continue
            if search_query:
                q = search_query.lower()
                if q not in item["prompt"].lower() and q not in item.get("seed", "").lower():
                    continue
            filtered.append(item)

        st.caption(f"Showing {len(filtered)} matching prompt(s)")

        for idx, item in enumerate(filtered):
            with st.expander(f"#{item['id']} [{item['type']}] · {item['created_at']} · {item.get('seed', '')[:60]}..."):
                st.code(item["prompt"], language=None)
                meta = item.get("metadata", {})
                if meta:
                    st.caption("Metadata: " + ", ".join(f"{k}: {v}" for k, v in meta.items()))
                st.download_button(
                    "Download this prompt",
                    data=item["prompt"],
                    file_name=f"prompt_{item['id']}.txt",
                    mime="text/plain",
                    key=f"dl_hist_{item['id']}_{idx}",
                )

        st.divider()

        export_c1, export_c2 = st.columns(2)
        with export_c1:
            all_text = "\n\n" + ("=" * 70) + "\n\n".join(
                f"[{item['type']}] - {item['created_at']}\nSeed: {item.get('seed', 'N/A')}\n\n{item['prompt']}"
                for item in history
            )
            st.download_button(
                "⬇️ Export Entire Library (.TXT)",
                data=all_text,
                file_name="prompt_library_export.txt",
                mime="text/plain",
                use_container_width=True,
            )

        with export_c2:
            all_json = json.dumps(history, indent=2)
            st.download_button(
                "⬇️ Export Entire Library (.JSON)",
                data=all_json,
                file_name="prompt_library_export.json",
                mime="application/json",
                use_container_width=True,
            )

        if st.button("🗑️ Clear Session History", type="secondary"):
            st.session_state.history = []
            st.session_state.current_image_results = []
            st.session_state.text_llm_result = ""
            st.rerun()


# ============================================================
# 5. MODEL DIAGNOSTICS & BENCHMARK
# ============================================================

elif page == "🔬 Model Diagnostics":
    st.markdown(
        """
        <div class="hero-box">
            <div class="hero-eyebrow">SYSTEM & TELEMETRY</div>
            <h1 class="hero-title">Model Diagnostics & Benchmark 🔬</h1>
            <p class="hero-desc">Detailed inspection of your local GPT-2 checkpoint weights, vocabulary, tensor architecture, and inference throughput.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not model_ready:
        st.error(f"Model failed to load: {model_error_msg}")
        st.stop()

    spec_c1, spec_c2, spec_c3 = st.columns(3)
    with spec_c1:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Model Architecture</div>
                <div class="stat-value">{model.__class__.__name__}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Total Parameters</div>
                <div class="stat-value">{model_meta.get('parameters', 0):,}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Embedding Dimension</div>
                <div class="stat-value">{model_meta.get('embedding_dim', 768)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with spec_c2:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Vocabulary Size</div>
                <div class="stat-value">{model_meta.get('vocab_size', 50257):,} tokens</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Transformer Layers</div>
                <div class="stat-value">{model_meta.get('layers', 12)} layers</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Max Context Positions</div>
                <div class="stat-value">{model_meta.get('context_window', 1024)} tokens</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with spec_c3:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Execution Device</div>
                <div class="stat-value">{model_meta.get('device', 'CPU')}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Weights Format</div>
                <div class="stat-value">{model_meta.get('model_type', 'safetensors')}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Model Load Latency</div>
                <div class="stat-value">{model_meta.get('load_time_sec', 0.0)} seconds</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader("Checkpoint Storage Location")
    st.code(model_meta.get("model_path", "N/A"), language=None)

    st.divider()

    st.subheader("⚡ Live Speed Benchmark")
    st.caption("Measure the inference speed of your local model by generating a standardized 50-token prompt.")

    if st.button("🚀 Run Inference Benchmark", type="primary"):
        with st.spinner("Benchmarking tokens per second..."):
            test_prompt = "a majestic futuristic city with flying vehicles, octane render"
            t0 = time.time()
            out = engine.generate_image_prompts(
                seed=test_prompt,
                tokenizer=tokenizer,
                model=model,
                device=device,
                num_variations=1,
                max_new_tokens=50,
            )
            duration = time.time() - t0
            tokens = out[0].get("tokens_generated", 50)
            tok_per_sec = tokens / max(duration, 0.001)

            b1, b2, b3 = st.columns(3)
            with b1:
                st.metric("Total Generation Time", f"{duration:.2f} s")
            with b2:
                st.metric("Tokens Generated", f"{tokens} tokens")
            with b3:
                st.metric("Throughput Speed", f"{tok_per_sec:.1f} tokens/sec")

            st.success("Benchmark completed successfully!")
            st.caption("Benchmark Output:")
            st.code(out[0]["prompt"], language=None)
