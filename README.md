# Cog Template Repository

This is a template repository for creating [Cog](https://github.com/replicate/cog) models that efficiently handle model weights with proper caching. It includes tools to upload model weights to Google Cloud Storage and generate download code for your `predict.py` file.

[![Replicate](https://replicate.com/zsxkib/model-name/badge)](https://replicate.com/zsxkib/model-name)

## Getting Started

To use this template for your own model:

1. Clone this repository
2. Modify `predict.py` with your model's implementation
3. Update `cog.yaml` with your model's dependencies
4. Use `cache_manager.py` to upload and manage model weights

## Repository Structure

- `predict.py`: The main model implementation file 
- `cache_manager.py`: Script for uploading model weights to GCS and generating download code
- `cog.yaml`: Cog configuration file that defines your model's environment

## Managing Model Weights with cache_manager.py

A key feature of this template is the `cache_manager.py` script, which helps you:

1. Upload model weights to Google Cloud Storage (GCS)
2. Generate code for downloading those weights in your `predict.py`
3. Handle both individual files and directories efficiently

### Prerequisites for Using cache_manager.py

- Google Cloud SDK installed and configured (`gcloud` command)
- Permission to upload to the specified GCS bucket (default: `gs://replicate-weights/`)
- `tar` command available in your PATH

### Basic Usage

```bash
python cache_manager.py --model-name your-model-name --local-dirs model_cache
```

This will:
1. Find files and directories in the `model_cache` directory
2. Create tar archives of each directory
3. Upload both individual files and tar archives to GCS
4. Generate code snippets for downloading the weights in your `predict.py`

### Advanced Usage

```bash
python cache_manager.py \
    --model-name your-model-name \
    --local-dirs model_cache weights \
    --gcs-base-path gs://replicate-weights/ \
    --cdn-base-url https://weights.replicate.delivery/default/ \
    --keep-tars
```

#### Parameters

- `--model-name`: Required. The name of your model (used in paths)
- `--local-dirs`: Required. One or more local directories to process
- `--gcs-base-path`: Optional. Base Google Cloud Storage path
- `--cdn-base-url`: Optional. Base CDN URL
- `--keep-tars`: Optional. Keep the generated .tar files locally after upload

## Workflow Example

1. **Develop your model locally**:
   ```bash
   # Run your model once to download weights to model_cache
   cog predict -i prompt="test"
   ```

2. **Upload model weights**:
   ```bash
   python cache_manager.py --model-name your-model-name --local-dirs model_cache
   ```

3. **Copy the generated code snippet** into your `predict.py`

4. **Test that the model can download weights**:
   ```bash
   rm -rf model_cache
   cog predict -i prompt="test"
   ```

## Example Implementation

The template comes with a sample Stable Diffusion implementation in `predict.py` that demonstrates:

- Setting up the model cache directory
- Downloading weights from GCS with progress reporting
- Setting environment variables for model caching
- Random seed generation for reproducibility
- Output format and quality options

## Best Practices

- **Environment Variables**: Set cache-related environment variables early
  ```python
  os.environ["HF_HOME"] = MODEL_CACHE
  os.environ["TORCH_HOME"] = MODEL_CACHE
  # etc.
  ```

- **Seed Management**: Provide a seed parameter and implement random seed generation
  ```python
  if seed is None:
      seed = int.from_bytes(os.urandom(2), "big")
  print(f"Using seed: {seed}")
  ```

- **Output Formats**: Support multiple output formats (webp, jpg, png) with quality controls
  ```python
  output_format: str = Input(
      description="Format of the output image",
      choices=["webp", "jpg", "png"],
      default="webp"
  )
  output_quality: int = Input(
      description="The image compression quality...",
      ge=1, le=100, default=80
  )
  ```

## Deploying to Replicate

After setting up your model, you can push it to [Replicate](https://replicate.com):

1. Create a new model on Replicate
2. Push your model:
   ```bash
   cog push r8.im/username/model-name
   ```

## License

MIT

---

---

⭐ Star this on [GitHub](https://github.com/zsxkib/model-name)!

👋 Follow `zsxkib` on [Twitter/X](https://twitter.com/zsakib_)
