"""
Compatibility patch for huggingface_hub.cached_download
This is needed because diffusers 0.31.0 requires cached_download which was removed in newer huggingface_hub versions.
"""
import sys
from huggingface_hub import hf_hub_download
from typing import Optional

def _cached_download_compat(
    url: str,
    cache_dir: Optional[str] = None,
    force_download: bool = False,
    proxies: Optional[dict] = None,
    resume_download: bool = False,
    local_files_only: bool = False,
    use_auth_token: Optional[bool] = None,
    **kwargs
):
    """
    Compatibility wrapper for cached_download using hf_hub_download.
    Note: This only works for HuggingFace Hub URLs, not arbitrary URLs.
    """
    # If it's a HuggingFace Hub URL, extract repo_id and filename
    if "huggingface.co" in url or url.startswith("hf://"):
        # Try to use hf_hub_download
        try:
            # Parse URL to get repo_id and filename
            if "/resolve/" in url:
                parts = url.split("/resolve/")
                repo_path = parts[0].replace("https://huggingface.co/", "").rstrip("/")
                file_path = parts[1].split("/")[-1] if "/" in parts[1] else parts[1]
                return hf_hub_download(
                    repo_id=repo_path,
                    filename=file_path,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    local_files_only=local_files_only,
                    token=use_auth_token if use_auth_token else None,
                )
        except Exception:
            pass
    
    # Fallback: Use urllib for direct URL downloads
    import urllib.request
    import tempfile
    import os
    from pathlib import Path
    
    if cache_dir:
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        # Create a filename from URL
        filename = url.split("/")[-1].split("?")[0]
        file_path = cache_path / filename
    else:
        file_path = tempfile.NamedTemporaryFile(delete=False).name
    
    if not force_download and os.path.exists(file_path):
        return str(file_path)
    
    # Download the file
    try:
        urllib.request.urlretrieve(url, file_path)
        return str(file_path)
    except Exception as e:
        raise RuntimeError(f"Failed to download {url}: {e}")

# Patch huggingface_hub if cached_download doesn't exist
try:
    from huggingface_hub import cached_download
except ImportError:
    import huggingface_hub
    huggingface_hub.cached_download = _cached_download_compat
    # Also add it to __all__ if it exists
    if hasattr(huggingface_hub, '__all__') and 'cached_download' not in huggingface_hub.__all__:
        huggingface_hub.__all__.append('cached_download')
