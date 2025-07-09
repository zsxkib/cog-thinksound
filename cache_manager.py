#!/usr/bin/env python3
# prepare_cog_cache.py

"""
HOW TO USE THIS SCRIPT
======================

This script helps you package model weights for Cog-based Replicate models. It:
1. Finds directories and files in your model cache folders
2. Creates .tar archives of each directory
3. Uploads both .tar archives and individual files to Google Cloud Storage
4. Generates code snippets for downloading the weights in predict.py

Basic Usage:
-----------
python cache_manager.py --model-name your-model --local-dirs model_cache weights

Required Arguments:
------------------
--model-name    : The name of your model (used in paths, e.g., test-sd-15)
--local-dirs    : One or more local directories to process (e.g., model_cache weights)

Optional Arguments:
------------------
--gcs-base-path : Base Google Cloud Storage path (default: gs://replicate-weights/)
--cdn-base-url  : Base CDN URL (default: https://weights.replicate.delivery/default/)
--keep-tars     : Keep the generated .tar files locally after upload (default: delete them)

Example Workflow:
----------------
1. Run your model once to download weights to ./model_cache:
   $ cog predict -i prompt="test"

2. Use this script to upload the model weights:
   $ python cache_manager.py \
       --model-name test-sd-15 \
       --local-dirs model_cache weights \
       --gcs-base-path gs://replicate-weights/ \
       --cdn-base-url https://weights.replicate.delivery/default/

3. Copy the generated code snippet into your predict.py

This will upload contents of:
   - ./model_cache to gs://replicate-weights/test-sd-15/model_cache/
   - ./weights to gs://replicate-weights/test-sd-15/weights/

Requirements:
------------
- Google Cloud SDK installed and configured (gcloud command)
- tar command available in PATH
- Permission to upload to the specified GCS bucket
"""

import os
import subprocess
import argparse
import sys
from typing import List, Tuple, Dict

# --- Configuration (Might be overridden by args) ---
DEFAULT_GCS_BASE_PATH = "gs://replicate-weights/"
DEFAULT_CDN_BASE_URL = "https://weights.replicate.delivery/default/"

def run_command(command: List[str], description: str, cwd: str = None) -> bool:
    """Runs a command, prints info, returns True on success."""
    print(f"\n🔄 RUNNING: {description}")
    print(f"📋 $ {' '.join(command)}")
    try:
        process = subprocess.run(
            command,
            check=True,  # Raise CalledProcessError on failure
            capture_output=True, # Capture stdout/stderr
            text=True,
            cwd=cwd # Run in specified directory if needed
        )
        if process.stdout: print(process.stdout)
        if process.stderr: print(f"⚠️  {process.stderr}") # Should be empty on success usually
        print(f"✅ SUCCESS: {description}")
        return True
    except FileNotFoundError:
         print(f"❌ ERROR: Command not found: '{command[0]}'. Is it installed and in PATH?")
         return False
    except subprocess.CalledProcessError as e:
        print(f"❌ ERROR: {description} failed with code {e.returncode}")
        if e.stdout: print(f"📝 Stdout:\n{e.stdout}")
        if e.stderr: print(f"⚠️ Stderr:\n{e.stderr}")
        return False
    except Exception as e:
        print(f"❌ ERROR: An unexpected error occurred during {description}: {e}")
        return False

def find_items_in_dir(parent_dir: str) -> Dict[str, List[str]]:
    """Finds both files and subdirectories within the parent_dir."""
    items = {"files": [], "dirs": []}
    
    if not os.path.isdir(parent_dir):
        print(f"❌ Error: Directory '{parent_dir}' not found.")
        return items
        
    for item in os.listdir(parent_dir):
        full_path = os.path.join(parent_dir, item)
        if os.path.isdir(full_path):
            items["dirs"].append(item)
        elif os.path.isfile(full_path):
            items["files"].append(item)
            
    return items

def generate_snippet(model_dirs: Dict[str, List[str]], model_name: str, cdn_base_url: str) -> str:
    """Generates the Python code snippet for predict.py with a simpler approach.
    
    Args:
        model_dirs: Dictionary mapping directory names to lists of files to download
        model_name: Name of the model (for URL paths)
        cdn_base_url: Base URL for CDN downloads
    """
    # Ensure trailing slash for base url
    cdn_base_url_with_slash = cdn_base_url.rstrip('/') + '/'
    
    # Collect files from the model_cache directory (primary directory)
    primary_dir = "model_cache"
    model_cache_files = []
    
    if primary_dir in model_dirs:
        model_cache_files = model_dirs[primary_dir]
    
    # Create imports and common setup code
    snippet = """
# --- Auto-generated Cog Cache Download Snippet ---
import os
import subprocess
import time
from cog import BasePredictor # Assuming BasePredictor is used

MODEL_CACHE = "model_cache"
BASE_URL = "{base_url}{model_name}/{primary_dir}/"

def download_weights(url: str, dest: str) -> None:
    start = time.time()
    print("[!] Initiating download from URL: ", url)
    print("[~] Destination path: ", dest)
    if ".tar" in dest:
        dest = os.path.dirname(dest)
    command = ["pget", "-vf" + ("x" if ".tar" in url else ""), url, dest]
    try:
        print(f"[~] Running command: {{' '.join(command)}}")
        subprocess.check_call(command, close_fds=False)
    except subprocess.CalledProcessError as e:
        print(
            f"[ERROR] Failed to download weights. Command '{{' '.join(e.cmd)}}' returned non-zero exit status {{e.returncode}}."
        )
        raise
    print("[+] Download completed in: ", time.time() - start, "seconds")

class Predictor(BasePredictor):
    def setup(self) -> None:
        \"\"\"Load the model into memory to make running multiple predictions efficient\"\"\"
        # Create model cache directory if it doesn't exist
        if not os.path.exists(MODEL_CACHE):
            os.makedirs(MODEL_CACHE)
            
        # Set environment variables for model caching
        os.environ["HF_HOME"] = MODEL_CACHE
        os.environ["TORCH_HOME"] = MODEL_CACHE
        os.environ["HF_DATASETS_CACHE"] = MODEL_CACHE
        os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE
        os.environ["HUGGINGFACE_HUB_CACHE"] = MODEL_CACHE
""".format(base_url=cdn_base_url_with_slash, model_name=model_name, primary_dir=primary_dir)
    
    # Add model_files list - just filenames, not full paths
    snippet += "\n        model_files = [\n"
    for filename in sorted(model_cache_files):
        snippet += f'            "{filename}",\n'
    snippet += "        ]\n"
    
    # Add download loop with simpler path handling
    snippet += """
        for model_file in model_files:
            url = BASE_URL + model_file
            filename = url.split("/")[-1]
            dest_path = os.path.join(MODEL_CACHE, filename)
            if not os.path.exists(dest_path.replace(".tar", "")):
                download_weights(url, dest_path)
                
        # Load the model
        model_id = "sd-legacy/stable-diffusion-v1-5"
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            cache_dir=MODEL_CACHE
        )
        self.pipe = self.pipe.to("cuda")
"""
    
    return snippet

def process_directory(
    local_dir: str, 
    model_name: str,
    gcs_base_path: str,
    cdn_base_url: str,
    keep_tars: bool,
) -> Dict[str, List[str]]:
    """Process a single local directory, uploading its contents to GCS.
    
    Returns:
        Dictionary with 'success' status and 'files' list of uploaded files
    """
    dirname = os.path.basename(os.path.normpath(local_dir))
    
    # Construct the GCS and CDN paths for this directory
    gcs_dir_path = os.path.join(gcs_base_path.rstrip('/'), model_name, dirname).rstrip('/') + '/'
    
    print(f"\n📂 Processing directory: {local_dir}")
    print(f"☁️  Uploading to GCS: {gcs_dir_path}")
    
    # Find files and subdirectories in this directory
    items = find_items_in_dir(local_dir)
    files = items["files"]
    subdirs = items["dirs"]
    
    if not subdirs and not files:
        print(f"⚠️  No files or directories found in '{local_dir}'. Nothing to upload.")
        return {'success': False, 'files': []}
    
    if files:
        print(f"\n🔍 Found {len(files)} files to process: {', '.join(files)}")
    if subdirs:
        print(f"🔍 Found {len(subdirs)} directories to process: {', '.join(subdirs)}")
    
    uploaded_files = []           # Regular files
    uploaded_tar_files = []       # Tarred directories
    local_tar_files_to_clean = []
    all_steps_succeeded = True
    
    # First process individual files
    if files:
        print("\n" + "=" * 80)
        print(f"📄 UPLOADING INDIVIDUAL FILES FROM {dirname}")
        print("=" * 80)
        
        for i, filename in enumerate(files, 1):
            local_file_path = os.path.join(local_dir, filename)
            gcs_file_path = gcs_dir_path + filename
            
            print(f"\n📋 PROCESSING FILE {i}/{len(files)}: {filename}")
            print("─" * 80)
            
            # Upload file directly to GCS
            upload_command = ["gcloud", "storage", "cp", local_file_path, gcs_file_path]
            if run_command(upload_command, f"Upload to GCS: {filename} → {gcs_dir_path}"):
                uploaded_files.append(filename)
            else:
                all_steps_succeeded = False
    
    # Then process directories (tar and upload)
    if subdirs:
        print("\n" + "=" * 80)
        print(f"📦 PACKAGING AND UPLOADING DIRECTORIES FROM {dirname}")
        print("=" * 80)
        
        for i, subdirname in enumerate(subdirs, 1):
            tar_filename = f"{subdirname}.tar"
            local_tar_path = os.path.join("/tmp", tar_filename)
            dir_to_tar_path = os.path.join(local_dir, subdirname)
            
            print(f"\n📋 PROCESSING DIRECTORY {i}/{len(subdirs)}: {subdirname}")
            print("─" * 80)
            
            # 1. Create Tar archive
            # Use -C to change directory *before* adding files, so paths inside tar are relative to dirname
            tar_command = ["tar", "-cvf", local_tar_path, "-C", local_dir, subdirname]
            if not run_command(tar_command, f"Create tar archive: {tar_filename}"):
                all_steps_succeeded = False
                continue # Skip upload if tar failed
            local_tar_files_to_clean.append(local_tar_path)
            
            # 2. Upload Tar to GCS
            gcs_tar_path = gcs_dir_path + tar_filename
            # Use explicit path to local tar file for upload command
            upload_command = ["gcloud", "storage", "cp", local_tar_path, gcs_tar_path]
            if not run_command(upload_command, f"Upload to GCS: {tar_filename} → {gcs_dir_path}"):
                all_steps_succeeded = False
                # Continue with other directories
                continue
            
            uploaded_tar_files.append(tar_filename) # Add relative tar filename
    
    # Cleanup local tar files unless requested otherwise
    if not keep_tars and local_tar_files_to_clean:
        print("\n" + "=" * 80)
        print("🧹 CLEANING UP")
        print("=" * 80)
        print("🗑️  Removing local .tar files...")
        for tar_file in local_tar_files_to_clean:
            try:
                os.remove(tar_file)
                print(f" ✓ Removed {tar_file}")
            except OSError as e:
                print(f" ⚠️  Warning: Failed to remove {tar_file}: {e}")
    elif keep_tars and local_tar_files_to_clean:
        print("\n🔒 Keeping local .tar files as requested.")
    
    # Combine both lists, ensuring no duplicates
    all_files_uploaded = list(set(uploaded_files + uploaded_tar_files))
    
    # Print a warning if we found duplicates
    if len(all_files_uploaded) < len(uploaded_files) + len(uploaded_tar_files):
        print("\n⚠️  Found duplicate filenames between individual files and directory tarballs.")
        print("⚠️  Deduplicating to avoid downloading the same file twice.")
    
    # Return results
    return {
        'success': all_steps_succeeded and len(all_files_uploaded) > 0,
        'files': all_files_uploaded,
    }

def main():
    parser = argparse.ArgumentParser(
        description="Upload files and directories from multiple local folders to Replicate's GCS bucket, "
                    "and generate a Python snippet for downloading them using pget in Cog."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Name of your model (e.g., test-sd-15). Used in GCS and CDN paths."
    )
    parser.add_argument(
        "--local-dirs",
        type=str,
        nargs='+',
        required=True,
        help="One or more local directories to process (e.g., model_cache weights)"
    )
    parser.add_argument(
        "--gcs-base-path",
        type=str,
        default=DEFAULT_GCS_BASE_PATH,
        help=f"Base GCS path (default: {DEFAULT_GCS_BASE_PATH})"
    )
    parser.add_argument(
        "--cdn-base-url",
        type=str,
        default=DEFAULT_CDN_BASE_URL,
        help=f"Base CDN URL (default: {DEFAULT_CDN_BASE_URL})"
    )
    parser.add_argument(
        "--keep-tars",
        action="store_true",
        help="Keep the generated .tar files locally after upload (default: delete them)."
    )
    
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("🚀 REPLICATE COG CACHE MANAGER")
    print("=" * 80)
    
    print(f"\n📋 Model Name: {args.model_name}")
    print(f"📋 Processing directories: {', '.join(args.local_dirs)}")
    print(f"📋 GCS Base Path: {args.gcs_base_path}")
    print(f"📋 CDN Base URL: {args.cdn_base_url}")
    
    # Track uploads from all directories
    results = {}
    all_succeeded = True
    
    # Process each directory
    for local_dir in args.local_dirs:
        result = process_directory(
            local_dir=local_dir,
            model_name=args.model_name,
            gcs_base_path=args.gcs_base_path,
            cdn_base_url=args.cdn_base_url,
            keep_tars=args.keep_tars,
        )
        
        # Store results by directory name
        dirname = os.path.basename(os.path.normpath(local_dir))
        results[dirname] = result["files"]
        
        if not result["success"]:
            all_succeeded = False
    
    # Generate download snippet if at least one directory had successful uploads
    has_uploads = any(len(files) > 0 for files in results.values())
    
    if has_uploads:
        print("\n" + "=" * 80)
        print("🔄 GENERATING CODE SNIPPET")
        print("=" * 80)
        
        print("✏️  Generating download code snippet...")
        
        snippet = generate_snippet(
            model_dirs=results,
            model_name=args.model_name,
            cdn_base_url=args.cdn_base_url,
        )
        
        print("\n" + "=" * 80)
        print("📝 PYTHON CODE SNIPPET FOR PREDICT.PY")
        print("=" * 80 + "\n")
        print(snippet)
        print("=" * 80)
        print("✅ END OF SNIPPET")
        print("=" * 80 + "\n")
    else:
        print("\n❌ No files were successfully uploaded. Skipping snippet generation.")
        all_succeeded = False
    
    print("\n" + "=" * 80)
    print("🏁 SUMMARY")
    print("=" * 80)
    
    if not all_succeeded:
        print("\n⚠️  Warning: One or more uploads failed during the process.")
        
    # Print summary of all uploads
    for dirname, files in results.items():
        if files:
            print(f"\n📂 Directory: {dirname}")
            print(f"  ✅ Successfully uploaded {len(files)} files")
            gcs_path = f"{args.gcs_base_path.rstrip('/')}/{args.model_name}/{dirname}/"
            cdn_url = f"{args.cdn_base_url.rstrip('/')}/{args.model_name}/{dirname}/"
            print(f"  📂 GCS Path: {gcs_path}")
            print(f"  🌐 CDN URL: {cdn_url}")
        else:
            print(f"\n📂 Directory: {dirname}")
            print(f"  ❌ No files were successfully uploaded")
    
    print("\n🔍 To verify uploads, run:")
    for dirname in results.keys():
        gcs_path = f"{args.gcs_base_path.rstrip('/')}/{args.model_name}/{dirname}/"
        print(f"  gsutil ls {gcs_path}")
    
    if not all_succeeded:
        sys.exit(1)
    else:
        print("\n✅ Process completed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()