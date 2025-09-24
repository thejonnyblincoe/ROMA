"""
Tests for VideoArtifact following Agno media patterns.

Tests cover all content sources (URL, file, bytes), validation,
serialization, and Agno-compatible methods.
"""

import base64
import pytest
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

from roma.domain.entities.artifacts.video_artifact import VideoArtifact
from roma.domain.value_objects.media_type import MediaType


class TestVideoArtifact:
    """Test VideoArtifact following Agno patterns."""

    def test_content_source_validation_success(self):
        """Test valid single content source initialization."""
        # Test with bytes content
        video_bytes = VideoArtifact(
            name="test_video",
            content=b"fake_video_data",
            mime_type="video/mp4"
        )
        assert video_bytes.content == b"fake_video_data"
        assert video_bytes.url is None
        assert video_bytes.filepath is None

        # Test with URL
        video_url = VideoArtifact(
            name="test_video",
            url="https://example.com/video.mp4",
            mime_type="video/mp4"
        )
        assert video_url.url == "https://example.com/video.mp4"
        assert video_url.content is None
        assert video_url.filepath is None

        # Test with filepath
        video_file = VideoArtifact(
            name="test_video",
            filepath="/path/to/video.mp4",
            mime_type="video/mp4"
        )
        assert video_file.filepath == Path("/path/to/video.mp4")
        assert video_file.content is None
        assert video_file.url is None

    def test_content_source_validation_failure(self):
        """Test validation fails with multiple or no content sources."""
        # Test with no content sources
        with pytest.raises(ValueError, match="Exactly one content source"):
            VideoArtifact(name="test_video", mime_type="video/mp4")

        # Test with multiple content sources
        with pytest.raises(ValueError, match="Exactly one content source"):
            VideoArtifact(
                name="test_video",
                content=b"data",
                url="https://example.com/video.mp4",
                mime_type="video/mp4"
            )

    def test_mime_type_validation(self):
        """Test MIME type validation."""
        # Valid MIME types
        valid_types = [
            "video/mp4", "video/avi", "video/mov", "video/wmv",
            "video/flv", "video/webm", "video/mkv", "video/m4v"
        ]

        for mime_type in valid_types:
            video = VideoArtifact(
                name="test",
                content=b"data",
                mime_type=mime_type
            )
            assert video.mime_type == mime_type

        # Invalid MIME type
        with pytest.raises(ValueError, match="Invalid video MIME type"):
            VideoArtifact(
                name="test",
                content=b"data",
                mime_type="audio/mpeg"
            )

    def test_filepath_conversion(self):
        """Test filepath string to Path conversion."""
        video = VideoArtifact(
            name="test",
            filepath="/path/to/video.mp4",
            mime_type="video/mp4"
        )
        assert isinstance(video.filepath, Path)
        assert str(video.filepath) == "/path/to/video.mp4"

    def test_media_type_property(self):
        """Test media_type property returns VIDEO."""
        video = VideoArtifact(
            name="test",
            content=b"data",
            mime_type="video/mp4"
        )
        assert video.media_type == MediaType.VIDEO

    @pytest.mark.asyncio
    async def test_get_content_from_bytes(self):
        """Test getting content from bytes."""
        content = b"fake_video_data"
        video = VideoArtifact(
            name="test",
            content=content,
            mime_type="video/mp4"
        )

        result = await video.get_content()
        assert result == content

    @pytest.mark.asyncio
    async def test_get_content_from_file(self):
        """Test getting content from file."""
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            test_content = b"fake_video_data"
            tmp.write(test_content)
            tmp.flush()

            video = VideoArtifact(
                name="test",
                filepath=tmp.name,
                mime_type="video/mp4"
            )

            result = await video.get_content()
            assert result == test_content

            # Cleanup
            Path(tmp.name).unlink()

    @pytest.mark.asyncio
    async def test_get_content_from_url(self):
        """Test getting content from URL."""
        mock_response = AsyncMock()
        mock_response.content = b"fake_video_data"
        mock_response.headers = {"content-type": "video/mp4"}
        mock_response.raise_for_status = AsyncMock()

        with patch('httpx.AsyncClient') as mock_client:
            mock_client_instance = AsyncMock()
            mock_client_instance.get.return_value = mock_response
            mock_client.return_value.__aenter__.return_value = mock_client_instance

            video = VideoArtifact(
                name="test",
                url="https://example.com/video.mp4",
                mime_type="video/mp4"
            )

            result = await video.get_content()
            assert result == b"fake_video_data"

    def test_get_content_summary(self):
        """Test content summary generation."""
        video = VideoArtifact(
            name="test_video",
            content=b"data",
            mime_type="video/mp4",
            format="MP4",
            duration_seconds=180.5,
            width=1920,
            height=1080,
            fps=30.0,
            bitrate=5000000
        )

        summary = video.get_content_summary()
        assert "Video: test_video" in summary
        assert "format=MP4" in summary
        assert "duration=3:00" in summary
        assert "resolution=1920x1080" in summary
        assert "fps=30.0" in summary
        assert "bitrate=5000000bps" in summary
        assert "source=bytes" in summary

    def test_get_size_bytes_from_content(self):
        """Test size calculation from bytes content."""
        content = b"fake_video_data"
        video = VideoArtifact(
            name="test",
            content=content,
            mime_type="video/mp4"
        )

        assert video.get_size_bytes() == len(content)

    def test_get_size_bytes_from_file(self):
        """Test size calculation from file."""
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            test_content = b"fake_video_data"
            tmp.write(test_content)
            tmp.flush()

            video = VideoArtifact(
                name="test",
                filepath=tmp.name,
                mime_type="video/mp4"
            )

            assert video.get_size_bytes() == len(test_content)

            # Cleanup
            Path(tmp.name).unlink()

    def test_is_accessible(self):
        """Test accessibility checking."""
        # Bytes content - always accessible
        video_bytes = VideoArtifact(
            name="test",
            content=b"data",
            mime_type="video/mp4"
        )
        assert video_bytes.is_accessible() is True

        # URL - assume accessible
        video_url = VideoArtifact(
            name="test",
            url="https://example.com/video.mp4",
            mime_type="video/mp4"
        )
        assert video_url.is_accessible() is True

        # Non-existent file - not accessible
        video_file = VideoArtifact(
            name="test",
            filepath="/non/existent/file.mp4",
            mime_type="video/mp4"
        )
        assert video_file.is_accessible() is False

    def test_get_file_extension_from_filepath(self):
        """Test file extension extraction from filepath."""
        video = VideoArtifact(
            name="test",
            filepath="/path/to/video.mp4",
            mime_type="video/mp4"
        )
        assert video.get_file_extension() == ".mp4"

    def test_get_file_extension_from_format(self):
        """Test file extension extraction from format."""
        video = VideoArtifact(
            name="test",
            content=b"data",
            mime_type="video/mp4",
            format="MP4"
        )
        assert video.get_file_extension() == ".mp4"

    def test_get_file_extension_from_mime_type(self):
        """Test file extension extraction from MIME type."""
        video = VideoArtifact(
            name="test",
            content=b"data",
            mime_type="video/avi"
        )
        assert video.get_file_extension() == ".avi"

    @pytest.mark.asyncio
    async def test_to_base64_without_data_url(self):
        """Test base64 encoding without data URL prefix."""
        content = b"fake_video_data"
        video = VideoArtifact(
            name="test",
            content=content,
            mime_type="video/mp4"
        )

        b64_str = await video.to_base64(include_data_url=False)
        expected = base64.b64encode(content).decode('utf-8')
        assert b64_str == expected

    @pytest.mark.asyncio
    async def test_to_base64_with_data_url(self):
        """Test base64 encoding with data URL prefix."""
        content = b"fake_video_data"
        video = VideoArtifact(
            name="test",
            content=content,
            mime_type="video/mp4"
        )

        b64_str = await video.to_base64(include_data_url=True)
        expected = f"data:video/mp4;base64,{base64.b64encode(content).decode('utf-8')}"
        assert b64_str == expected

    def test_from_base64_plain(self):
        """Test creation from plain base64 string."""
        content = b"fake_video_data"
        b64_str = base64.b64encode(content).decode('utf-8')

        video = VideoArtifact.from_base64(
            base64_str=b64_str,
            name="test_video",
            mime_type="video/mp4"
        )

        assert video.name == "test_video"
        assert video.content == content
        assert video.mime_type == "video/mp4"
        assert video.url is None
        assert video.filepath is None

    def test_from_base64_data_url(self):
        """Test creation from data URL format."""
        content = b"fake_video_data"
        b64_str = base64.b64encode(content).decode('utf-8')
        data_url = f"data:video/avi;base64,{b64_str}"

        video = VideoArtifact.from_base64(
            base64_str=data_url,
            name="test_video"
        )

        assert video.name == "test_video"
        assert video.content == content
        assert video.mime_type == "video/avi"  # Extracted from data URL

    def test_from_base64_invalid(self):
        """Test creation from invalid base64 string."""
        with pytest.raises(ValueError, match="Invalid base64 data"):
            VideoArtifact.from_base64(
                base64_str="invalid_base64_data",
                name="test_video"
            )

    def test_from_url(self):
        """Test creation from URL."""
        video = VideoArtifact.from_url(
            url="https://example.com/video.mp4",
            name="test_video",
            mime_type="video/mp4",
            duration_seconds=120.0,
            width=1920,
            height=1080
        )

        assert video.name == "test_video"
        assert video.url == "https://example.com/video.mp4"
        assert video.mime_type == "video/mp4"
        assert video.duration_seconds == 120.0
        assert video.width == 1920
        assert video.height == 1080
        assert video.content is None
        assert video.filepath is None

    def test_from_file_with_name(self):
        """Test creation from file with explicit name."""
        video = VideoArtifact.from_file(
            filepath="/path/to/video.mp4",
            name="custom_name",
            mime_type="video/mp4"
        )

        assert video.name == "custom_name"
        assert video.filepath == Path("/path/to/video.mp4")
        assert video.mime_type == "video/mp4"

    def test_from_file_auto_name(self):
        """Test creation from file with auto-detected name."""
        video = VideoArtifact.from_file(
            filepath="/path/to/video.mp4"
        )

        assert video.name == "video.mp4"
        assert video.filepath == Path("/path/to/video.mp4")
        assert video.mime_type == "video/mp4"  # Auto-detected

    def test_from_file_auto_mime_type(self):
        """Test creation from file with auto-detected MIME type."""
        test_cases = [
            ("/path/to/video.mp4", "video/mp4"),
            ("/path/to/video.avi", "video/avi"),
            ("/path/to/video.mov", "video/mov"),
            ("/path/to/video.wmv", "video/wmv"),
            ("/path/to/video.flv", "video/flv"),
            ("/path/to/video.webm", "video/webm"),
            ("/path/to/video.mkv", "video/mkv"),
            ("/path/to/video.m4v", "video/m4v"),
            ("/path/to/video.unknown", "video/mp4"),  # Default
        ]

        for filepath, expected_mime in test_cases:
            video = VideoArtifact.from_file(filepath=filepath)
            assert video.mime_type == expected_mime

    def test_to_dict_basic(self):
        """Test dictionary serialization without content."""
        video = VideoArtifact(
            name="test_video",
            url="https://example.com/video.mp4",
            mime_type="video/mp4",
            format="MP4",
            duration_seconds=180.0,
            width=1920,
            height=1080,
            fps=30.0,
            bitrate=5000000
        )

        result = video.to_dict(include_content=False)

        assert result["name"] == "test_video"
        assert result["url"] == "https://example.com/video.mp4"
        assert result["filepath"] is None
        assert result["format"] == "MP4"
        assert result["duration_seconds"] == 180.0
        assert result["width"] == 1920
        assert result["height"] == 1080
        assert result["fps"] == 30.0
        assert result["bitrate"] == 5000000
        assert "content_base64" not in result

    def test_to_dict_with_content(self):
        """Test dictionary serialization with content."""
        content = b"fake_video_data"
        video = VideoArtifact(
            name="test_video",
            content=content,
            mime_type="video/mp4"
        )

        result = video.to_dict(include_content=True)

        expected_b64 = base64.b64encode(content).decode('utf-8')
        assert result["content_base64"] == expected_b64

    def test_from_dict_with_content_base64(self):
        """Test creation from dictionary with base64 content."""
        content = b"fake_video_data"
        b64_content = base64.b64encode(content).decode('utf-8')

        data = {
            "name": "test_video",
            "mime_type": "video/mp4",
            "format": "MP4",
            "duration_seconds": 180.0,
            "width": 1920,
            "height": 1080,
            "content_base64": b64_content,
            "url": "https://example.com/should_be_cleared.mp4",  # Should be cleared
            "filepath": "/path/should_be_cleared.mp4"  # Should be cleared
        }

        video = VideoArtifact.from_dict(data)

        assert video.name == "test_video"
        assert video.content == content
        assert video.mime_type == "video/mp4"
        assert video.format == "MP4"
        assert video.duration_seconds == 180.0
        assert video.width == 1920
        assert video.height == 1080
        assert video.url is None  # Cleared due to content_base64 priority
        assert video.filepath is None  # Cleared due to content_base64 priority

    def test_from_dict_with_filepath_string(self):
        """Test creation from dictionary with filepath as string."""
        data = {
            "name": "test_video",
            "mime_type": "video/mp4",
            "filepath": "/path/to/video.mp4"
        }

        video = VideoArtifact.from_dict(data)

        assert video.name == "test_video"
        assert video.filepath == Path("/path/to/video.mp4")
        assert isinstance(video.filepath, Path)

    def test_immutability(self):
        """Test that VideoArtifact is immutable."""
        video = VideoArtifact(
            name="test",
            content=b"data",
            mime_type="video/mp4"
        )

        # Should not be able to modify fields
        with pytest.raises(Exception):  # ValidationError or AttributeError
            video.name = "modified"

        with pytest.raises(Exception):  # ValidationError or AttributeError
            video.content = b"modified_data"

    def test_duration_formatting_in_summary(self):
        """Test various duration formatting in content summary."""
        test_cases = [
            (30.0, "0:30"),
            (65.5, "1:05"),
            (3600.0, "60:00"),
            (3725.8, "62:05")
        ]

        for duration, expected in test_cases:
            video = VideoArtifact(
                name="test",
                content=b"data",
                mime_type="video/mp4",
                duration_seconds=duration
            )

            summary = video.get_content_summary()
            assert f"duration={expected}" in summary

    def test_resolution_formatting_in_summary(self):
        """Test resolution formatting in content summary."""
        video = VideoArtifact(
            name="test",
            content=b"data",
            mime_type="video/mp4",
            width=1920,
            height=1080
        )

        summary = video.get_content_summary()
        assert "resolution=1920x1080" in summary

    def test_source_identification_in_summary(self):
        """Test source identification in content summary."""
        # URL source
        video_url = VideoArtifact(
            name="test",
            url="https://example.com/video.mp4",
            mime_type="video/mp4"
        )
        assert "source=URL" in video_url.get_content_summary()

        # File source
        video_file = VideoArtifact(
            name="test",
            filepath="/path/to/test.mp4",
            mime_type="video/mp4"
        )
        assert "source=file(test.mp4)" in video_file.get_content_summary()

        # Bytes source
        video_bytes = VideoArtifact(
            name="test",
            content=b"data",
            mime_type="video/mp4"
        )
        assert "source=bytes" in video_bytes.get_content_summary()