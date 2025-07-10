# ThinkSound: Step-by-step reasoning for video-to-audio generation

[![Replicate](https://replicate.com/zsxkib/thinksound/badge)](https://replicate.com/zsxkib/thinksound)

This repository contains a Cog implementation of ThinkSound, a video-to-audio generation model that thinks through what sounds should happen in your videos. This Cog package makes it easy to run ThinkSound locally or deploy it to Replicate.

> **⚠️ For research and educational use only. No commercial use.**

ThinkSound doesn't just match sounds to objects like other models. Instead, it thinks through what sounds should happen and when, creating natural audio that fits the mood, timing, and context of your video. It's like having an AI sound designer that watches your video and creates a complete audio track that fits perfectly.

Original research:
*   Original Project: [liuhuadai/ThinkSound](https://github.com/liuhuadai/ThinkSound)
*   Research Paper: [ThinkSound: Chain-of-Thought Reasoning in Multimodal Large Language Models for Audio Generation and Editing](https://arxiv.org/abs/2506.21448)
*   Model Weights: [FunAudioLLM/ThinkSound](https://huggingface.co/FunAudioLLM/ThinkSound)
*   This Cog implementation: [zsxkib on GitHub](https://github.com/zsxkib)

## Prerequisites

*   **Docker**: [Install Docker](https://docs.docker.com/get-docker/)
*   **Cog**: [Install Cog](https://github.com/replicate/cog#install)
*   **NVIDIA GPU**: At least 16GB memory (A100, H100, or RTX 4090+ recommended)

## Quick start

1. **Clone this repository:**
   ```bash
   git clone https://github.com/zsxkib/cog-thinksound.git
   cd cog-thinksound
   ```

2. **Run predictions:**
   
   Basic usage:
   ```bash
   # Generate audio from video
   cog predict -i video=@your_video.mp4
   
   # Add a caption for better results
   cog predict -i video=@cooking.mp4 -i caption="Cooking pasta in a kitchen"
   ```
   
   Advanced usage with step-by-step reasoning:
   ```bash
   # Detailed audio description for precise control
   cog predict \
     -i video=@rain.mp4 \
     -i caption="Rain on window" \
     -i cot="Begin with gentle raindrops hitting glass, gradually building to steady rainfall. Add subtle ambient sounds of water flowing and distant thunder."
   ```
   
   Professional controls:
   ```bash
   # High quality generation
   cog predict \
     -i video=@nature.mp4 \
     -i caption="Forest wildlife" \
     -i cot="Create layered nature sounds with bird calls, rustling leaves, and distant animal sounds" \
     -i num_inference_steps=50 \
     -i cfg_scale=7.0
   
   # Fast generation
   cog predict \
     -i video=@test.mp4 \
     -i caption="Quick test" \
     -i num_inference_steps=15
   
   # Reproducible results
   cog predict \
     -i video=@demo.mp4 \
     -i caption="Demo video" \
     -i seed=42
   ```

## Parameters

- **video**: Input video file (MP4, AVI, MOV, etc.)
- **caption**: Brief description of video content (optional but recommended)
- **cot**: Detailed step-by-step description of desired audio (optional)
- **cfg_scale** (1.0-20.0, default 5.0): How closely to follow text descriptions
- **num_inference_steps** (10-100, default 24): Quality vs speed tradeoff
- **seed**: Random seed for reproducible outputs (leave empty for random)

## Deploy your own version

Push your own version to Replicate:

```bash
cog login
cog push r8.im/your-username/thinksound
```

## How this works

This Cog package includes optimizations that make it faster and easier to use:

- **Dual-GPU support**: Uses multiple GPUs when available for faster processing
- **Smart caching**: Reuses model weights between runs so you don't wait as long
- **Video processing**: Efficient frame extraction at multiple rates
- **Fine-tuning controls**: Adjust quality, creativity, and reproducibility
- **Memory management**: Handles large models efficiently without running out of memory

## Technical details

The implementation processes video at two frame rates:
- 8fps for understanding what's happening in the video
- 25fps for precise timing between audio and video

Text processing uses multiple encoders:
- MetaCLIP for connecting visual content with text descriptions
- T5 for understanding detailed step-by-step reasoning

Audio generation uses a process with configurable steps and guidance strength.

## License and usage

> **Important**: This model is **for research and educational purposes only**.  
> **Commercial use is NOT permitted** without explicit licensing from the original authors.

This Cog implementation follows the original ThinkSound project's Apache 2.0 license for the code, but the model weights and research are subject to non-commercial restrictions.

For commercial licensing, contact the original research team.

## Support

- Original ThinkSound project: [github.com/liuhuadai/ThinkSound](https://github.com/liuhuadai/ThinkSound)
- This Cog implementation: [github.com/zsxkib/cog-thinksound](https://github.com/zsxkib/cog-thinksound)
- Issues with this Cog package: [Open an issue](https://github.com/zsxkib/cog-thinksound/issues)

---

⭐ Star this repo if you find it useful!

🐦 Follow [@zsakib_](https://twitter.com/zsakib_) for updates