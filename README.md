# PromptCraft AI Studio 🎨✨

A modern, high-performance web interface for your fine-tuned **GPT-2 Image Prompt Generation Model** (trained on DiffusionDB).

---

## 🚀 Quick Start

Run the application from the project root using either `uv` or standard Python:

```powershell
# Using uv (recommended)
uv run streamlit run app.py

# Or using the local virtual environment directly
.\.venv\Scripts\streamlit.exe run app.py
```

Open your browser at **`http://localhost:8501`**.

---

## 🧠 Model Checkpoint Details

The application automatically discovers and loads your local trained checkpoint with zero runtime downloads:

- **Location**: `model/trained_model/trained_model/`
- **Architecture**: `GPT2LMHeadModel` (Causal Language Model)
- **Parameters**: `124,439,808` (~124.4M)
- **Format**: `model.safetensors` + `config.json` + `tokenizer.json`
- **Fallback Formats**: Also supports `model/image_prompt_model.pth` and `model/image_prompt_model.pkl`.
- **Hardware Acceleration**: Automatic GPU (`CUDA`) detection with fallback to multi-threaded `CPU`.

---

## 🌟 Key Features & Studio Modules

### 1. 🎨 Image Prompt Studio
- **Seed Expansion**: Enter a short idea (e.g. *"cyberpunk samurai in rainy neon city"*) and let the model enrich it into a production-grade generative art prompt.
- **Inspiration Chips**: 1-click starter prompts (*Cyberpunk Samurai, Cosmic Jellyfish, Steampunk Forest, Floating Castle, Android Portrait, Crystal Dragon*).
- **Artistic Modifiers**:
  - *Styles*: Photorealistic, Cinematic, Anime/Manga, 3D Octane Render, Fantasy Concept Art, Dark Fantasy, Watercolor, Surrealism, etc.
  - *Lighting*: Volumetric God Rays, Cyberpunk Neon, Golden Hour, Bioluminescent Glow, Studio Softbox.
  - *Camera & Framing*: Wide Angle, Macro Close-up, Low Angle Hero, 85mm Bokeh, Drone Aerial.
  - *Boosters*: ArtStation Trending, Unreal Engine 5, Octane 3D, CGSociety Contest Winner, Greg Rutkowski & WLOP.
- **Target Platform Formatting**:
  - **Midjourney v6**: Adds `/imagine prompt:` with `--ar` and `--v 6.0` flags.
  - **Stable Diffusion / SDXL**: Formats positive prompt weights and suggests an automatic **Negative Prompt**.
  - **Flux.1 & DALL-E 3**: Formats for descriptive, natural-language visual comprehension.
- **Batch Variations**: Generate 1 to 4 distinct variations in parallel.
- **Tag Inspector**: Automatic breakdown pills for detected styles, 3D engines, lighting, and camera settings.

### 2. ⚖️ A/B Model Comparison
- Test two configurations side-by-side on the exact same seed concept.
- Compare low temperature (focused/deterministic) vs high temperature (wild/creative) or test different aesthetic styles.

### 3. ✍️ LLM Prompt Architect
- Structured prompt engineering workbench for language models (**ChatGPT, Claude, Gemini**).
- Customize Role, Objective, Audience, Tone, Format, Depth, and Specific Constraints.

### 4. 📚 Prompt Library & Session History
- Catalogs all prompts generated in your active session.
- Search and filter by prompt type or keyword.
- 1-click export to **`.TXT`** and **`.JSON`**.

### 5. 🔬 Model Diagnostics & Speed Benchmark
- Live parameter breakdown, embedding dimensions (`768`), layer count (`12`), context window (`1024`), and vocabulary (`50,257`).
- **Live Throughput Benchmark**: Measures real-time tokens-per-second generation speed.

---

## 📁 Project Structure

```text
prompt/
├── app.py                      # Main Streamlit web application
├── engine.py                   # Centralized model loader, inference & tag extractor
├── requirements.txt            # Python dependencies (streamlit, torch, transformers, safetensors)
├── pyproject.toml              # uv project definition
├── .streamlit/
│   └── config.toml             # Sleek dark-mode studio theme configuration
├── model/                      # Your local trained model files
│   ├── trained_model/
│   │   └── trained_model/      # Safetensors checkpoint + tokenizer (primary)
│   ├── image_prompt_model.pth  # PyTorch weights backup
│   ├── image_prompt_model.pkl  # Pickle weights backup
│   └── tokenizer/
│       └── tokenizer/          # Tokenizer config & vocab
└── UI/
    ├── app.py                  # Standalone UI wrapper
    └── predict.py              # Modular predict interface
```
