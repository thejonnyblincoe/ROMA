"""
Tests for AudioArtifact following Agno media patterns.

Tests cover all content sources (URL, file, bytes), validation,
serialization, and Agno-compatible methods.
"""

import base64
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from roma.domain.entities.artifacts.audio_artifact import AudioArtifact
from roma.domain.value_objects.media_type import MediaType


class TestAudioArtifact:
    """Test AudioArtifact following Agno patterns."""

    def test_content_source_validation_success(self):
        """Test valid single content source initialization."""
        # Test with bytes content
        audio_bytes = AudioArtifact(
            name="test_audio",
            content=b"fake_audio_data",
            mime_type="audio/mpeg"
        )
        assert audio_bytes.content == b"fake_audio_data"
        assert audio_bytes.url is None
        assert audio_bytes.filepath is None

        # Test with URL
        audio_url = AudioArtifact(
            name="test_audio",
            url="https://example.com/audio.mp3",
            mime_type="audio/mpeg"
        )
        assert audio_url.url == "https://example.com/audio.mp3"
        assert audio_url.content is None
        assert audio_url.filepath is None

        # Test with filepath
        audio_file = AudioArtifact(
            name="test_audio",
            filepath="/path/to/audio.mp3",
            mime_type="audio/mpeg"
        )
        assert audio_file.filepath == Path("/path/to/audio.mp3")
        assert audio_file.content is None
        assert audio_file.url is None

    def test_content_source_validation_failure(self):
        """Test validation fails with multiple or no content sources."""
        # Test with no content sources
        with pytest.raises(ValueError, match="Exactly one content source"):
            AudioArtifact(name="test_audio", mime_type="audio/mpeg")

        # Test with multiple content sources
        with pytest.raises(ValueError, match="Exactly one content source"):
            AudioArtifact(
                name="test_audio",
                content=b"data",
                url="https://example.com/audio.mp3",
                mime_type="audio/mpeg"
            )

    def test_mime_type_validation(self):
        """Test MIME type validation."""
        # Valid MIME types
        valid_types = [
            "audio/mpeg", "audio/mp3", "audio/wav", "audio/flac",
            "audio/ogg", "audio/aac", "audio/m4a", "audio/webm"
        ]

        for mime_type in valid_types:
            audio = AudioArtifact(
                name="test",
                content=b"data",
                mime_type=mime_type
            )
            assert audio.mime_type == mime_type

        # Invalid MIME type
        with pytest.raises(ValueError, match="Invalid audio MIME type"):
            AudioArtifact(
                name="test",
                content=b"data",
                mime_type="video/mp4"
            )

    def test_filepath_conversion(self):
        """Test filepath string to Path conversion."""
        audio = AudioArtifact(
            name="test",
            filepath="/path/to/audio.mp3",
            mime_type="audio/mpeg"
        )
        assert isinstance(audio.filepath, Path)
        assert str(audio.filepath) == "/path/to/audio.mp3"

    def test_media_type_property(self):
        """Test media_type property returns AUDIO."""
        audio = AudioArtifact(
            name="test",
            content=b"data",
            mime_type="audio/mpeg"
        )
        assert audio.media_type == MediaType.AUDIO

    @pytest.mark.asyncio
    async def test_get_content_from_bytes(self):
        """Test getting content from bytes."""
        content = b"fake_audio_data"
        audio = AudioArtifact(
            name="test",
            content=content,
            mime_type="audio/mpeg"
        )

        result = await audio.get_content()
        assert result == content

    @pytest.mark.asyncio
    async def test_get_content_from_file(self):
        """Test getting content from file."""
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp:
            test_content = b"fake_audio_data"
            tmp.write(test_content)
            tmp.flush()

            audio = AudioArtifact(
                name="test",
                filepath=tmp.name,
                mime_type="audio/mpeg"
            )

            result = await audio.get_content()
            assert result == test_content

            # Cleanup
            Path(tmp.name).unlink()

    @pytest.mark.asyncio
    async def test_get_content_from_url(self):
        """Test getting content from URL."""
        mock_response = AsyncMock()
        mock_response.content = b"fake_audio_data"
        mock_response.headers = {"content-type": "audio/mpeg"}
        mock_response.raise_for_status = AsyncMock()

        with patch('httpx.AsyncClient') as mock_client:
            mock_client_instance = AsyncMock()
            mock_client_instance.get.return_value = mock_response
            mock_client.return_value.__aenter__.return_value = mock_client_instance

            audio = AudioArtifact(
                name="test",
                url="https://example.com/audio.mp3",
                mime_type="audio/mpeg"
            )

            result = await audio.get_content()
            assert result == b"fake_audio_data"

    def test_get_content_summary(self):
        """Test content summary generation."""
        audio = AudioArtifact(
            name="test_audio",
            content=b"data",
            mime_type="audio/mpeg",
            format="MP3",
            duration_seconds=180.5,
            sample_rate=44100,
            channels=2
        )

        summary = audio.get_content_summary()
        assert "Audio: test_audio" in summary
        assert "format=MP3" in summary
        assert "duration=3:00" in summary
        assert "rate=44100Hz" in summary
        assert "2ch" in summary
        assert "source=bytes" in summary

    def test_get_size_bytes_from_content(self):
        """Test size calculation from bytes content."""
        content = b"fake_audio_data"
        audio = AudioArtifact(
            name="test",
            content=content,
            mime_type="audio/mpeg"
        )

        assert audio.get_size_bytes() == len(content)

    def test_get_size_bytes_from_file(self):
        """Test size calculation from file."""
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp:
            test_content = b"fake_audio_data"
            tmp.write(test_content)
            tmp.flush()

            audio = AudioArtifact(
                name="test",
                filepath=tmp.name,
                mime_type="audio/mpeg"
            )

            assert audio.get_size_bytes() == len(test_content)

            # Cleanup
            Path(tmp.name).unlink()

    def test_is_accessible(self):
        """Test accessibility checking."""
        # Bytes content - always accessible
        audio_bytes = AudioArtifact(
            name="test",
            content=b"data",
            mime_type="audio/mpeg"
        )
        assert audio_bytes.is_accessible() is True

        # URL - assume accessible
        audio_url = AudioArtifact(
            name="test",
            url="https://example.com/audio.mp3",
            mime_type="audio/mpeg"
        )
        assert audio_url.is_accessible() is True

        # Non-existent file - not accessible
        audio_file = AudioArtifact(
            name="test",
            filepath="/non/existent/file.mp3",
            mime_type="audio/mpeg"
        )
        assert audio_file.is_accessible() is False

    def test_get_file_extension_from_filepath(self):
        """Test file extension extraction from filepath."""
        audio = AudioArtifact(
            name="test",
            filepath="/path/to/audio.mp3",
            mime_type="audio/mpeg"
        )
        assert audio.get_file_extension() == ".mp3"

    def test_get_file_extension_from_format(self):
        """Test file extension extraction from format."""
        audio = AudioArtifact(
            name="test",
            content=b"data",
            mime_type="audio/mpeg",
            format="MP3"
        )
        assert audio.get_file_extension() == ".mp3"

    def test_get_file_extension_from_mime_type(self):
        """Test file extension extraction from MIME type."""
        audio = AudioArtifact(
            name="test",
            content=b"data",
            mime_type="audio/flac"
        )
        assert audio.get_file_extension() == ".flac"

    @pytest.mark.asyncio
    async def test_to_base64_without_data_url(self):
        """Test base64 encoding without data URL prefix."""
        content = b"fake_audio_data"
        audio = AudioArtifact(
            name="test",
            content=content,
            mime_type="audio/mpeg"
        )

        b64_str = await audio.to_base64(include_data_url=False)
        expected = base64.b64encode(content).decode('utf-8')
        assert b64_str == expected

    @pytest.mark.asyncio
    async def test_to_base64_with_data_url(self):
        """Test base64 encoding with data URL prefix."""
        content = b"fake_audio_data"
        audio = AudioArtifact(
            name="test",
            content=content,
            mime_type="audio/mpeg"
        )

        b64_str = await audio.to_base64(include_data_url=True)
        expected = f"data:audio/mpeg;base64,{base64.b64encode(content).decode('utf-8')}"
        assert b64_str == expected

    def test_from_base64_plain(self):
        """Test creation from plain base64 string."""
        content = b"fake_audio_data"
        b64_str = base64.b64encode(content).decode('utf-8')

        audio = AudioArtifact.from_base64(
            base64_str=b64_str,
            name="test_audio",
            mime_type="audio/mpeg"
        )

        assert audio.name == "test_audio"
        assert audio.content == content
        assert audio.mime_type == "audio/mpeg"
        assert audio.url is None
        assert audio.filepath is None

    def test_from_base64_data_url(self):
        """Test creation from data URL format."""
        content = b"fake_audio_data"
        b64_str = base64.b64encode(content).decode('utf-8')
        data_url = f"data:audio/flac;base64,{b64_str}"

        audio = AudioArtifact.from_base64(
            base64_str=data_url,
            name="test_audio"
        )

        assert audio.name == "test_audio"
        assert audio.content == content
        assert audio.mime_type == "audio/flac"  # Extracted from data URL

    def test_from_base64_invalid(self):
        """Test creation from invalid base64 string."""
        with pytest.raises(ValueError, match="Invalid base64 data"):
            AudioArtifact.from_base64(
                base64_str="invalid_base64_data",
                name="test_audio"
            )

    def test_from_url(self):
        """Test creation from URL."""
        audio = AudioArtifact.from_url(
            url="https://example.com/audio.mp3",
            name="test_audio",
            mime_type="audio/mpeg",
            duration_seconds=120.0
        )

        assert audio.name == "test_audio"
        assert audio.url == "https://example.com/audio.mp3"
        assert audio.mime_type == "audio/mpeg"
        assert audio.duration_seconds == 120.0
        assert audio.content is None
        assert audio.filepath is None

    def test_from_file_with_name(self):
        """Test creation from file with explicit name."""
        audio = AudioArtifact.from_file(
            filepath="/path/to/audio.mp3",
            name="custom_name",
            mime_type="audio/mpeg"
        )

        assert audio.name == "custom_name"
        assert audio.filepath == Path("/path/to/audio.mp3")
        assert audio.mime_type == "audio/mpeg"

    def test_from_file_auto_name(self):
        """Test creation from file with auto-detected name."""
        audio = AudioArtifact.from_file(
            filepath="/path/to/audio.mp3"
        )

        assert audio.name == "audio.mp3"
        assert audio.filepath == Path("/path/to/audio.mp3")
        assert audio.mime_type == "audio/mpeg"  # Auto-detected

    def test_from_file_auto_mime_type(self):
        """Test creation from file with auto-detected MIME type."""
        test_cases = [
            ("/path/to/audio.mp3", "audio/mpeg"),
            ("/path/to/audio.wav", "audio/wav"),
            ("/path/to/audio.flac", "audio/flac"),
            ("/path/to/audio.ogg", "audio/ogg"),
            ("/path/to/audio.aac", "audio/aac"),
            ("/path/to/audio.m4a", "audio/m4a"),
            ("/path/to/audio.webm", "audio/webm"),
            ("/path/to/audio.unknown", "audio/mpeg"),  # Default
        ]

        for filepath, expected_mime in test_cases:
            audio = AudioArtifact.from_file(filepath=filepath)
            assert audio.mime_type == expected_mime

    def test_to_dict_basic(self):
        """Test dictionary serialization without content."""
        audio = AudioArtifact(
            name="test_audio",
            url="https://example.com/audio.mp3",
            mime_type="audio/mpeg",
            format="MP3",
            duration_seconds=180.0,
            sample_rate=44100,
            channels=2
        )

        result = audio.to_dict(include_content=False)

        assert result["name"] == "test_audio"
        assert result["url"] == "https://example.com/audio.mp3"
        assert result["filepath"] is None
        assert result["format"] == "MP3"
        assert result["duration_seconds"] == 180.0
        assert result["sample_rate"] == 44100
        assert result["channels"] == 2
        assert "content_base64" not in result

    def test_to_dict_with_content(self):
        """Test dictionary serialization with content."""
        content = b"fake_audio_data"
        audio = AudioArtifact(
            name="test_audio",
            content=content,
            mime_type="audio/mpeg"
        )

        result = audio.to_dict(include_content=True)

        expected_b64 = base64.b64encode(content).decode('utf-8')
        assert result["content_base64"] == expected_b64

    def test_from_dict_with_content_base64(self):
        """Test creation from dictionary with base64 content."""
        content = b"fake_audio_data"
        b64_content = base64.b64encode(content).decode('utf-8')

        data = {
            "name": "test_audio",
            "mime_type": "audio/mpeg",
            "format": "MP3",
            "duration_seconds": 180.0,
            "content_base64": b64_content,
            "url": "https://example.com/should_be_cleared.mp3",  # Should be cleared
            "filepath": "/path/should_be_cleared.mp3"  # Should be cleared
        }

        audio = AudioArtifact.from_dict(data)

        assert audio.name == "test_audio"
        assert audio.content == content
        assert audio.mime_type == "audio/mpeg"
        assert audio.format == "MP3"
        assert audio.duration_seconds == 180.0
        assert audio.url is None  # Cleared due to content_base64 priority
        assert audio.filepath is None  # Cleared due to content_base64 priority

    def test_from_dict_with_filepath_string(self):
        """Test creation from dictionary with filepath as string."""
        data = {
            "name": "test_audio",
            "mime_type": "audio/mpeg",
            "filepath": "/path/to/audio.mp3"
        }

        audio = AudioArtifact.from_dict(data)

        assert audio.name == "test_audio"
        assert audio.filepath == Path("/path/to/audio.mp3")
        assert isinstance(audio.filepath, Path)

    def test_immutability(self):
        """Test that AudioArtifact is immutable."""
        audio = AudioArtifact(
            name="test",
            content=b"data",
            mime_type="audio/mpeg"
        )

        # Should not be able to modify fields
        with pytest.raises(Exception):  # ValidationError or AttributeError
            audio.name = "modified"

        with pytest.raises(Exception):  # ValidationError or AttributeError
            audio.content = b"modified_data"

    def test_duration_formatting_in_summary(self):
        """Test various duration formatting in content summary."""
        test_cases = [
            (30.0, "0:30"),
            (65.5, "1:05"),
            (3600.0, "60:00"),
            (3725.8, "62:05")
        ]

        for duration, expected in test_cases:
            audio = AudioArtifact(
                name="test",
                content=b"data",
                mime_type="audio/mpeg",
                duration_seconds=duration
            )

            summary = audio.get_content_summary()
            assert f"duration={expected}" in summary
