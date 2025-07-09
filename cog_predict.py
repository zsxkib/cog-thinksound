#!/usr/bin/env python3
"""
ThinkSound: Chain-of-Thought Reasoning in Multimodal Large Language Models for Audio Generation and Editing

Prediction interface for Cog ⚙️
https://cog.run/python

This module provides a Replicate-compatible predictor for ThinkSound, a unified Any2Audio generation 
framework with flow matching guided by Chain-of-Thought (CoT) reasoning. It can generate high-quality 
audio from videos with optional text conditioning.

Author: ThinkSound Team
License: Apache 2.0
Repository: https://github.com/liuhuadai/ThinkSound
Paper: https://arxiv.org/abs/2506.21448
"""

import os
import json
import torch
import torchaudio
import tempfile
import subprocess
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from cog import BasePredictor, Input, Path as CogPath

from lightning.pytorch import seed_everything

from ThinkSound.models import create_model_from_config
from ThinkSound.models.utils import load_ckpt_state_dict
from ThinkSound.inference.sampling import sample, sample_discrete_euler
from data_utils.v2a_utils.feature_utils_224 import FeaturesUtils
from torchvision.transforms import v2
from torio.io import StreamingMediaDecoder
from transformers import AutoProcessor
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from moviepy.editor import VideoFileClip

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Video processing constants
_CLIP_SIZE = 224
_CLIP_FPS = 8.0
_SYNC_SIZE = 224
_SYNC_FPS = 25.0

# Model configuration constants
DEFAULT_SEED = 10086
DEFAULT_SAMPLE_RATE = 44100
CFG_SCALE = 5.0
NUM_INFERENCE_STEPS = 24


def pad_to_square(video_tensor: torch.Tensor) -> torch.Tensor:
    """
    Pad video tensor to square dimensions by adding padding to the smaller dimension.
    
    Args:
        video_tensor: Input tensor with shape (length, channels, height, width)
        
    Returns:
        Padded tensor with square spatial dimensions
    """
    if len(video_tensor.shape) != 4:
        raise ValueError(f"Input tensor must have shape (l, c, h, w), got {video_tensor.shape}")

    l, c, h, w = video_tensor.shape
    max_side = max(h, w)

    pad_h = max_side - h
    pad_w = max_side - w
    
    # Calculate symmetric padding
    padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
    video_padded = F.pad(video_tensor, pad=padding, mode='constant', value=0)

    return video_padded


class VideoPreprocessor:
    """
    Video preprocessing class for ThinkSound model.
    
    Handles video loading, frame extraction at multiple frame rates, and feature preparation
    for both CLIP (8fps) and synchronization (25fps) encoders.
    """

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        duration_sec: float = 9.0,
        audio_samples: Optional[int] = None,
        normalize_audio: bool = False,
    ):
        """
        Initialize video preprocessor.
        
        Args:
            sample_rate: Audio sample rate in Hz
            duration_sec: Expected video duration in seconds
            audio_samples: Override for exact number of audio samples
            normalize_audio: Whether to normalize audio (unused in current implementation)
        """
        if audio_samples is None:
            self.audio_samples = int(sample_rate * duration_sec)
        else:
            self.audio_samples = audio_samples
            effective_duration = audio_samples / sample_rate
            # Ensure duration consistency within 15ms tolerance
            assert abs(effective_duration - duration_sec) < 0.015, \
                f'audio_samples {audio_samples} does not match duration_sec {duration_sec}'

        self.sample_rate = sample_rate
        self.duration_sec = duration_sec

        # Calculate expected sequence lengths
        self.expected_audio_length = self.audio_samples
        self.clip_expected_length = int(_CLIP_FPS * self.duration_sec)
        self.sync_expected_length = int(_SYNC_FPS * self.duration_sec)

        # Initialize video transforms
        self._setup_transforms()

    def _setup_transforms(self) -> None:
        """Setup video transformation pipelines for CLIP and sync encoders."""
        
        # CLIP video transform (square, 224x224)
        self.clip_transform = v2.Compose([
            v2.Lambda(pad_to_square),
            v2.Resize((_CLIP_SIZE, _CLIP_SIZE), interpolation=v2.InterpolationMode.BICUBIC),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ])
        
        # Initialize CLIP processor
        self.clip_processor = AutoProcessor.from_pretrained("facebook/metaclip-h14-fullcc2.5b")
        
        # Sync video transform (224x224, center crop)
        self.sync_transform = v2.Compose([
            v2.Resize(_SYNC_SIZE, interpolation=v2.InterpolationMode.BICUBIC),
            v2.CenterCrop(_SYNC_SIZE),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def process_video(self, video_path: str, caption: str, cot: str) -> Dict[str, Any]:
        """
        Process video file and extract features for model input.
        
        Args:
            video_path: Path to input video file
            caption: Text caption for the video
            cot: Chain-of-Thought description
            
        Returns:
            Dictionary containing processed video features and metadata
        """
        logger.info(f"Processing video: {video_path}")
        
        # Setup video reader with dual stream processing
        reader = StreamingMediaDecoder(video_path)
        
        # Add CLIP video stream (8fps)
        reader.add_basic_video_stream(
            frames_per_chunk=int(_CLIP_FPS * self.duration_sec),
            frame_rate=_CLIP_FPS,
            format='rgb24',
        )
        
        # Add sync video stream (25fps) 
        reader.add_basic_video_stream(
            frames_per_chunk=int(_SYNC_FPS * self.duration_sec),
            frame_rate=_SYNC_FPS,
            format='rgb24',
        )

        # Extract video frames
        reader.fill_buffer()
        data_chunk = reader.pop_chunks()

        clip_chunk = data_chunk[0]
        sync_chunk = data_chunk[1]

        if sync_chunk is None:
            raise RuntimeError(f'Failed to extract sync video stream from {video_path}')

        # Process CLIP frames
        clip_chunk = self._process_clip_frames(clip_chunk)
        
        # Process sync frames  
        sync_chunk = self._process_sync_frames(sync_chunk)

        return {
            'id': video_path,
            'caption': caption,
            'caption_cot': cot,
            'clip_video': clip_chunk,
            'sync_video': sync_chunk,
        }

    def _process_clip_frames(self, clip_chunk: torch.Tensor) -> torch.Tensor:
        """Process CLIP video frames with padding if necessary."""
        clip_chunk = clip_chunk[:self.clip_expected_length]
        
        if clip_chunk.shape[0] != self.clip_expected_length:
            clip_chunk = self._pad_frames(
                clip_chunk, 
                self.clip_expected_length, 
                max_padding=4, 
                stream_name="CLIP"
            )
        
        # Apply transforms and CLIP processing
        clip_chunk = pad_to_square(clip_chunk)
        clip_chunk = self.clip_processor(images=clip_chunk, return_tensors="pt")["pixel_values"]
        
        return clip_chunk

    def _process_sync_frames(self, sync_chunk: torch.Tensor) -> torch.Tensor:
        """Process sync video frames with padding if necessary."""
        sync_chunk = sync_chunk[:self.sync_expected_length]
        
        if sync_chunk.shape[0] != self.sync_expected_length:
            sync_chunk = self._pad_frames(
                sync_chunk, 
                self.sync_expected_length, 
                max_padding=12, 
                stream_name="sync"
            )
        
        # Apply sync transforms
        sync_chunk = self.sync_transform(sync_chunk)
        
        return sync_chunk

    def _pad_frames(
        self, 
        frames: torch.Tensor, 
        target_length: int, 
        max_padding: int, 
        stream_name: str
    ) -> torch.Tensor:
        """Pad frame sequence to target length by repeating last frame."""
        current_length = frames.shape[0]
        padding_needed = target_length - current_length
        
        if padding_needed > max_padding:
            raise RuntimeError(
                f'{stream_name} stream padding exceeds maximum allowed ({max_padding}). '
                f'Needed: {padding_needed}, got length: {current_length}, target: {target_length}'
            )
        
        if padding_needed > 0:
            last_frame = frames[-1]
            padding = last_frame.repeat(padding_needed, 1, 1, 1)
            frames = torch.cat((frames, padding), dim=0)
            
        return frames


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds using moviepy."""
    with VideoFileClip(video_path) as video:
        return video.duration


class Predictor(BasePredictor):
    """
    ThinkSound model predictor for Replicate.
    
    This predictor implements the full ThinkSound pipeline:
    1. Video preprocessing and feature extraction
    2. Multi-modal conditioning (text, video)
    3. Diffusion-based audio generation
    4. Audio-video combination
    """

    def setup(self) -> None:
        """
        Load the ThinkSound model and initialize all components.
        
        This method downloads model weights, sets up feature extractors,
        and prepares the diffusion model for inference.
        """
        logger.info("Setting up ThinkSound predictor...")
        
        # Configure devices
        self._setup_devices()
        
        # Set reproducible seed
        seed_everything(DEFAULT_SEED, workers=True)
        logger.info(f"Set random seed to {DEFAULT_SEED}")

        # Download and load models
        self._download_model_weights()
        self._load_feature_extractor()
        self._load_diffusion_model()
        
        logger.info("✅ ThinkSound model setup complete!")

    def _setup_devices(self) -> None:
        """Configure GPU/CPU devices for model inference."""
        if torch.cuda.is_available():
            self.device = 'cuda'
            self.extra_device = 'cuda:1' if torch.cuda.device_count() > 1 else 'cuda:0'
            logger.info(f"Using GPU devices: primary={self.device}, secondary={self.extra_device}")
        else:
            self.device = 'cpu'
            self.extra_device = 'cpu'
            logger.warning("CUDA not available, using CPU (will be slow)")

    def _download_model_weights(self) -> None:
        """Download model checkpoints from HuggingFace."""
        logger.info("Downloading model checkpoints from HuggingFace...")
        
        self.vae_ckpt = hf_hub_download(
            repo_id="FunAudioLLM/ThinkSound", 
            filename="vae.ckpt", 
            repo_type="model"
        )
        
        self.synchformer_ckpt = hf_hub_download(
            repo_id="FunAudioLLM/ThinkSound", 
            filename="synchformer_state_dict.pth", 
            repo_type="model"
        )
        
        self.diffusion_ckpt = hf_hub_download(
            repo_id="FunAudioLLM/ThinkSound", 
            filename="thinksound_light.ckpt", 
            repo_type="model"
        )
        
        logger.info("Model checkpoints downloaded successfully")

    def _load_feature_extractor(self) -> None:
        """Initialize and load the multi-modal feature extractor."""
        logger.info("Loading feature extractor...")
        
        self.feature_extractor = FeaturesUtils(
            vae_ckpt=None,
            vae_config='ThinkSound/configs/model_configs/stable_audio_2_0_vae.json',
            enable_conditions=True,
            synchformer_ckpt=self.synchformer_ckpt
        ).eval().to(self.extra_device)
        
        logger.info("Feature extractor loaded successfully")

    def _load_diffusion_model(self) -> None:
        """Load and configure the diffusion model."""
        logger.info("Loading diffusion model...")
        
        # Load model configuration
        config_path = "ThinkSound/configs/model_configs/thinksound.json"
        with open(config_path) as f:
            model_config = json.load(f)

        # Create and load diffusion model
        self.diffusion_model = create_model_from_config(model_config)
        state_dict = torch.load(self.diffusion_ckpt, map_location='cpu')
        self.diffusion_model.load_state_dict(state_dict)
        self.diffusion_model.to(self.device)

        # Load VAE weights
        logger.info("Loading VAE weights...")
        vae_state = load_ckpt_state_dict(self.vae_ckpt, prefix='autoencoder.')
        self.diffusion_model.pretransform.load_state_dict(vae_state)

        logger.info("Diffusion model loaded successfully")

    def predict(
        self,
        video: CogPath = Input(description="Input video file (supports various formats)"),
        caption: str = Input(
            description="Caption/title describing the video content (optional)", 
            default=""
        ),
        cot: str = Input(
            description="Chain-of-Thought description providing detailed reasoning about the desired audio (optional)", 
            default=""
        ),
    ) -> CogPath:
        """
        Generate audio from video using ThinkSound.
        
        This method processes the input video, extracts multi-modal features,
        generates audio using diffusion sampling, and combines the result with
        the original video.
        
        Args:
            video: Input video file path
            caption: Brief description of the video content
            cot: Detailed Chain-of-Thought description for audio generation
            
        Returns:
            Path to the output video with generated audio
        """
        logger.info("🎬 Starting ThinkSound audio generation...")
        
        # Prepare inputs
        video_path = str(video)
        caption = caption.strip() if caption else ''
        cot = cot.strip() if cot else caption
        
        # Get video properties
        duration_sec = get_video_duration(video_path)
        logger.info(f"📹 Video duration: {duration_sec:.2f} seconds")
        
        # Process video and extract features
        features = self._extract_features(video_path, caption, cot, duration_sec)
        
        # Generate audio using diffusion
        audio_tensor = self._generate_audio(features, duration_sec)
        
        # Combine audio with video
        output_path = self._combine_audio_video(video_path, audio_tensor)
        
        logger.info("✅ Audio generation completed successfully!")
        return CogPath(output_path)

    def _extract_features(
        self, 
        video_path: str, 
        caption: str, 
        cot: str, 
        duration_sec: float
    ) -> Dict[str, Any]:
        """Extract multi-modal features from video and text."""
        logger.info("🔍 Extracting multi-modal features...")
        
        # Process video
        preprocessor = VideoPreprocessor(duration_sec=duration_sec)
        video_data = preprocessor.process_video(video_path, caption, cot)

        # Extract text features
        logger.info("Processing text features...")
        metaclip_global_features, metaclip_text_features = self.feature_extractor.encode_text(
            video_data['caption']
        )
        t5_features = self.feature_extractor.encode_t5_text(video_data['caption_cot'])

        # Extract video features
        logger.info("Processing video features...")
        clip_features = self.feature_extractor.encode_video_with_clip(
            video_data['clip_video'].unsqueeze(0).to(self.extra_device)
        )
        sync_features = self.feature_extractor.encode_video_with_sync(
            video_data['sync_video'].unsqueeze(0).to(self.extra_device)
        )

        # Prepare feature dictionary
        features = {
            'metaclip_global_text_features': metaclip_global_features.detach().cpu().squeeze(0),
            'metaclip_text_features': metaclip_text_features.detach().cpu().squeeze(0),
            't5_features': t5_features.detach().cpu().squeeze(0),
            'metaclip_features': clip_features.detach().cpu().squeeze(0),
            'sync_features': sync_features.detach().cpu().squeeze(0),
            'video_exist': torch.tensor(True)
        }

        logger.info(f"📊 Extracted features - CLIP: {features['metaclip_features'].shape}, "
                   f"Sync: {features['sync_features'].shape}")
        
        return features

    def _generate_audio(self, features: Dict[str, Any], duration_sec: float) -> torch.Tensor:
        """Generate audio using the diffusion model."""
        logger.info("🎵 Generating audio with diffusion model...")
        
        # Calculate sequence lengths
        sync_seq_len = features['sync_features'].shape[0]
        clip_seq_len = features['metaclip_features'].shape[0]
        latent_seq_len = int(194 / 9 * duration_sec)
        
        # Update model sequence lengths
        self.diffusion_model.model.model.update_seq_lengths(
            latent_seq_len, clip_seq_len, sync_seq_len
        )

        # Prepare conditioning
        metadata = [features]
        batch_size = 1
        
        with torch.amp.autocast(self.device):
            conditioning = self.diffusion_model.conditioner(metadata, self.device)
        
        # Handle video existence mask
        video_exist = torch.stack([item['video_exist'] for item in metadata], dim=0)
        conditioning['metaclip_features'][~video_exist] = \
            self.diffusion_model.model.model.empty_clip_feat
        conditioning['sync_features'][~video_exist] = \
            self.diffusion_model.model.model.empty_sync_feat

        # Generate audio latents
        cond_inputs = self.diffusion_model.get_conditioning_inputs(conditioning)
        noise = torch.randn([batch_size, self.diffusion_model.io_channels, latent_seq_len]).to(self.device)
        
        logger.info(f"Running {NUM_INFERENCE_STEPS} diffusion steps...")
        with torch.amp.autocast(self.device):
            if self.diffusion_model.diffusion_objective == "v":
                latents = sample(
                    self.diffusion_model.model, noise, NUM_INFERENCE_STEPS, 0, 
                    **cond_inputs, cfg_scale=CFG_SCALE, batch_cfg=True
                )
            elif self.diffusion_model.diffusion_objective == "rectified_flow":
                latents = sample_discrete_euler(
                    self.diffusion_model.model, noise, NUM_INFERENCE_STEPS, 
                    **cond_inputs, cfg_scale=CFG_SCALE, batch_cfg=True
                )
            else:
                raise ValueError(f"Unknown diffusion objective: {self.diffusion_model.diffusion_objective}")
                
            # Decode latents to audio
            if self.diffusion_model.pretransform is not None:
                audio_tensor = self.diffusion_model.pretransform.decode(latents)
            else:
                audio_tensor = latents

        # Normalize audio
        audio_tensor = audio_tensor.to(torch.float32)
        audio_tensor = audio_tensor.div(torch.max(torch.abs(audio_tensor))).clamp(-1, 1)
        
        return audio_tensor

    def _combine_audio_video(self, video_path: str, audio_tensor: torch.Tensor) -> str:
        """Combine generated audio with original video using FFmpeg."""
        logger.info("🎞️ Combining audio with video...")
        
        # Convert audio tensor to int16 and save
        audio_int16 = audio_tensor.mul(32767).to(torch.int16).cpu()
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_audio:
            torchaudio.save(tmp_audio.name, audio_int16[0], DEFAULT_SAMPLE_RATE)
            audio_path = tmp_audio.name

        # Create output video with combined audio
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_video:
            output_path = tmp_video.name

        # Use FFmpeg to combine video and audio
        ffmpeg_cmd = [
            'ffmpeg', '-y', '-i', video_path, '-i', audio_path,
            '-c:v', 'copy', '-map', '0:v:0', '-map', '1:a:0',
            '-shortest', output_path
        ]
        
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
        logger.info("Video and audio combined successfully")
        
        # Clean up temporary audio file
        os.unlink(audio_path)
        
        return output_path
