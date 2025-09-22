"""
Tests for VideoArtifact entity.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from pydantic import ValidationError

from roma.domain.entities.artifacts.video_artifact import VideoArtifact
from roma.domain.entities.media_file import MediaFile
from roma.domain.value_objects.media_type import MediaType


class TestVideoArtifact:
    """Test VideoArtifact entity."""

    @pytest.fixture
    def mock_media_file(self):
        """Create mock media file."""
        media_file = Mock(spec=MediaFile)
        media_file.name = "test_video.mp4"
        media_file.size = 10240000
        media_file.format = "video/mp4"
        media_file.get_content_bytes = AsyncMock(return_value=b"video_data")
        media_file.get_content_summary = Mock(return_value="Video file: test_video.mp4 (10.0 MB)")
        media_file.is_accessible = Mock(return_value=True)
        return media_file

    def test_create_video_artifact(self, mock_media_file):
        """Test creating video artifact."""
        artifact = VideoArtifact(
            name="Test Video",
            media_file=mock_media_file,
            duration_seconds=300.5,
            width=1920,
            height=1080,
            fps=30.0,
            task_id="test_task",
            metadata={"source": "camera"}
        )

        assert artifact.name == "Test Video"
        assert artifact.media_file == mock_media_file
        assert artifact.duration_seconds == 300.5
        assert artifact.width == 1920
        assert artifact.height == 1080
        assert artifact.fps == 30.0
        assert artifact.task_id == "test_task"
        assert artifact.metadata["source"] == "camera"

    def test_media_type_property(self, mock_media_file):
        """Test media type property returns VIDEO."""
        artifact = VideoArtifact(
            name="Test Video",
            media_file=mock_media_file
        )

        assert artifact.media_type == MediaType.VIDEO

    def test_model_post_init_validation(self):
        """Test validation fails without media file."""
        with pytest.raises(ValidationError):
            VideoArtifact(
                name="Test Video",
                media_file=None
            )

    def test_from_path_class_method(self):
        """Test creating video artifact from file path."""
        with patch('roma.domain.entities.media_file.MediaFile.from_filepath') as mock_from_filepath:
            mock_media_file = Mock(spec=MediaFile)
            mock_from_filepath.return_value = mock_media_file

            artifact = VideoArtifact.from_path(
                name="Video from path",
                file_path="/path/to/video.avi",
                duration_seconds=240.0,
                width=1280,
                height=720,
                fps=24.0,
                task_id="task_456",
                metadata={"quality": "hd"}
            )

            mock_from_filepath.assert_called_once_with("/path/to/video.avi", name="Video from path")
            assert artifact.name == "Video from path"
            assert artifact.media_file == mock_media_file
            assert artifact.duration_seconds == 240.0
            assert artifact.width == 1280
            assert artifact.height == 720
            assert artifact.fps == 24.0
            assert artifact.task_id == "task_456"
            assert artifact.metadata["quality"] == "hd"

    def test_from_url_class_method(self):
        """Test creating video artifact from URL."""
        with patch('roma.domain.entities.media_file.MediaFile.from_url') as mock_from_url:
            mock_media_file = Mock(spec=MediaFile)
            mock_from_url.return_value = mock_media_file

            artifact = VideoArtifact.from_url(
                name="Video from URL",
                file_url="https://example.com/video.mov",
                duration_seconds=600.0,
                width=3840,
                height=2160
            )

            mock_from_url.assert_called_once_with("https://example.com/video.mov", name="Video from URL")
            assert artifact.name == "Video from URL"
            assert artifact.media_file == mock_media_file
            assert artifact.duration_seconds == 600.0
            assert artifact.width == 3840
            assert artifact.height == 2160

    def test_from_bytes_class_method(self):
        """Test creating video artifact from bytes."""
        with patch('roma.domain.entities.media_file.MediaFile.from_bytes') as mock_from_bytes:
            mock_media_file = Mock(spec=MediaFile)
            mock_from_bytes.return_value = mock_media_file

            video_data = b"fake_video_data"
            artifact = VideoArtifact.from_bytes(
                name="Video from bytes",
                content=video_data,
                format="video/webm",
                duration_seconds=90.0,
                fps=25.0
            )

            mock_from_bytes.assert_called_once_with(video_data, name="Video from bytes", format="video/webm")
            assert artifact.name == "Video from bytes"
            assert artifact.media_file == mock_media_file
            assert artifact.duration_seconds == 90.0
            assert artifact.fps == 25.0

    @pytest.mark.asyncio
    async def test_get_content(self, mock_media_file):
        """Test getting content as bytes."""
        artifact = VideoArtifact(
            name="Test Video",
            media_file=mock_media_file
        )

        content = await artifact.get_content()

        mock_media_file.get_content_bytes.assert_called_once()
        assert content == b"video_data"

    def test_get_content_summary_full_info(self, mock_media_file):
        """Test getting content summary with all video info."""
        artifact = VideoArtifact(
            name="Test Video",
            media_file=mock_media_file,
            duration_seconds=300.5,
            width=1920,
            height=1080,
            fps=30.0
        )

        summary = artifact.get_content_summary()

        mock_media_file.get_content_summary.assert_called_once()
        assert "Video:" in summary
        assert "(300.5s)" in summary
        assert "1920x1080" in summary
        assert "@30.0fps" in summary

    def test_get_content_summary_partial_info(self, mock_media_file):
        """Test getting content summary with partial video info."""
        artifact = VideoArtifact(
            name="Test Video",
            media_file=mock_media_file,
            duration_seconds=120.0,
            width=1280,
            height=720
            # No fps
        )

        summary = artifact.get_content_summary()

        assert "Video:" in summary
        assert "(120.0s)" in summary
        assert "1280x720" in summary
        assert "fps" not in summary

    def test_get_content_summary_minimal_info(self, mock_media_file):
        """Test getting content summary with minimal video info."""
        artifact = VideoArtifact(
            name="Test Video",
            media_file=mock_media_file
            # No duration, width, height, or fps
        )

        summary = artifact.get_content_summary()

        assert "Video:" in summary
        assert "s)" not in summary  # No duration
        assert "x" not in summary  # No resolution
        assert "fps" not in summary  # No fps

    def test_get_size_bytes(self, mock_media_file):
        """Test getting size in bytes."""
        artifact = VideoArtifact(
            name="Test Video",
            media_file=mock_media_file
        )

        size = artifact.get_size_bytes()

        assert size == 10240000

    def test_is_accessible(self, mock_media_file):
        """Test accessibility check."""
        artifact = VideoArtifact(
            name="Test Video",
            media_file=mock_media_file
        )

        accessible = artifact.is_accessible()

        mock_media_file.is_accessible.assert_called_once()
        assert accessible is True

    def test_get_mime_type(self, mock_media_file):
        """Test getting MIME type."""
        artifact = VideoArtifact(
            name="Test Video",
            media_file=mock_media_file
        )

        mime_type = artifact.get_mime_type()

        assert mime_type == "video/mp4"

    def test_get_file_extension(self, mock_media_file):
        """Test getting file extension."""
        artifact = VideoArtifact(
            name="Test Video",
            media_file=mock_media_file
        )

        extension = artifact.get_file_extension()

        assert extension == ".mp4"

    def test_get_file_extension_no_extension(self, mock_media_file):
        """Test getting file extension when name has no extension."""
        mock_media_file.name = "video_file_no_extension"

        artifact = VideoArtifact(
            name="Test Video",
            media_file=mock_media_file
        )

        extension = artifact.get_file_extension()

        assert extension is None

    def test_get_file_extension_no_name(self, mock_media_file):
        """Test getting file extension when media file has no name."""
        mock_media_file.name = None

        artifact = VideoArtifact(
            name="Test Video",
            media_file=mock_media_file
        )

        extension = artifact.get_file_extension()

        assert extension is None

    def test_video_specific_fields(self, mock_media_file):
        """Test video-specific fields are properly stored."""
        artifact = VideoArtifact(
            name="Test Video",
            media_file=mock_media_file,
            duration_seconds=450.25,
            width=2560,
            height=1440,
            fps=60.0
        )

        # Test that all video-specific fields are accessible
        assert artifact.duration_seconds == 450.25
        assert artifact.width == 2560
        assert artifact.height == 1440
        assert artifact.fps == 60.0

    def test_optional_video_fields(self, mock_media_file):
        """Test that video-specific fields are optional."""
        artifact = VideoArtifact(
            name="Test Video",
            media_file=mock_media_file
        )

        # All video-specific fields should be None when not provided
        assert artifact.duration_seconds is None
        assert artifact.width is None
        assert artifact.height is None
        assert artifact.fps is None