"""
Artifacts package for ROMA v2.0 Multimodal Context.

Contains artifact classes for different media types.
Text is handled as plain strings - only non-text content uses artifacts.
"""

from .base_artifact import BaseArtifact
from .file_artifact import FileArtifact
from .image_artifact import ImageArtifact
from .audio_artifact import AudioArtifact
from .video_artifact import VideoArtifact

__all__ = [
    "BaseArtifact",
    "FileArtifact",
    "ImageArtifact",
    "AudioArtifact",
    "VideoArtifact",
]