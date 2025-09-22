"""
Tests for AudioArtifact entity.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from pydantic import ValidationError

from roma.domain.entities.artifacts.audio_artifact import AudioArtifact
from roma.domain.entities.media_file import MediaFile
from roma.domain.value_objects.media_type import MediaType


class TestAudioArtifact:
    """Test AudioArtifact entity."""

    @pytest.fixture
    def mock_media_file(self):
        """Create mock media file."""
        media_file = Mock(spec=MediaFile)
        media_file.name = "test_audio.mp3"
        media_file.size = 1024000
        media_file.format = "audio/mpeg"
        media_file.get_content_bytes = AsyncMock(return_value=b"audio_data")
        media_file.get_content_summary = Mock(return_value="Audio file: test_audio.mp3 (1.0 MB)")
        media_file.is_accessible = Mock(return_value=True)
        return media_file

    def test_create_audio_artifact(self, mock_media_file):
        """Test creating audio artifact."""
        artifact = AudioArtifact(
            name="Test Audio",
            media_file=mock_media_file,
            duration_seconds=120.5,
            task_id="test_task",
            metadata={"source": "microphone"}
        )

        assert artifact.name == "Test Audio"
        assert artifact.media_file == mock_media_file
        assert artifact.duration_seconds == 120.5
        assert artifact.task_id == "test_task"
        assert artifact.metadata["source"] == "microphone"

    def test_media_type_property(self, mock_media_file):
        """Test media type property returns AUDIO."""
        artifact = AudioArtifact(
            name="Test Audio",
            media_file=mock_media_file
        )

        assert artifact.media_type == MediaType.AUDIO

    def test_model_post_init_validation(self):
        """Test validation fails without media file."""
        with pytest.raises(ValidationError):
            AudioArtifact(
                name="Test Audio",
                media_file=None
            )

    def test_from_path_class_method(self):
        """Test creating audio artifact from file path."""
        with patch('roma.domain.entities.media_file.MediaFile.from_filepath') as mock_from_filepath:
            mock_media_file = Mock(spec=MediaFile)
            mock_from_filepath.return_value = mock_media_file

            artifact = AudioArtifact.from_path(
                name="Audio from path",
                file_path="/path/to/audio.wav",
                duration_seconds=60.0,
                task_id="task_123",
                metadata={"quality": "high"}
            )

            mock_from_filepath.assert_called_once_with("/path/to/audio.wav", name="Audio from path")
            assert artifact.name == "Audio from path"
            assert artifact.media_file == mock_media_file
            assert artifact.duration_seconds == 60.0
            assert artifact.task_id == "task_123"
            assert artifact.metadata["quality"] == "high"

    def test_from_url_class_method(self):
        """Test creating audio artifact from URL."""
        with patch('roma.domain.entities.media_file.MediaFile.from_url') as mock_from_url:
            mock_media_file = Mock(spec=MediaFile)
            mock_from_url.return_value = mock_media_file

            artifact = AudioArtifact.from_url(
                name="Audio from URL",
                file_url="https://example.com/audio.mp3",
                duration_seconds=180.0
            )

            mock_from_url.assert_called_once_with("https://example.com/audio.mp3", name="Audio from URL")
            assert artifact.name == "Audio from URL"
            assert artifact.media_file == mock_media_file
            assert artifact.duration_seconds == 180.0

    def test_from_bytes_class_method(self):
        """Test creating audio artifact from bytes."""
        with patch('roma.domain.entities.media_file.MediaFile.from_bytes') as mock_from_bytes:
            mock_media_file = Mock(spec=MediaFile)
            mock_from_bytes.return_value = mock_media_file

            audio_data = b"fake_audio_data"
            artifact = AudioArtifact.from_bytes(
                name="Audio from bytes",
                content=audio_data,
                format="audio/wav",
                duration_seconds=45.0
            )

            mock_from_bytes.assert_called_once_with(audio_data, name="Audio from bytes", format="audio/wav")
            assert artifact.name == "Audio from bytes"
            assert artifact.media_file == mock_media_file
            assert artifact.duration_seconds == 45.0

    @pytest.mark.asyncio
    async def test_get_content(self, mock_media_file):
        """Test getting content as bytes."""
        artifact = AudioArtifact(
            name="Test Audio",
            media_file=mock_media_file
        )

        content = await artifact.get_content()

        mock_media_file.get_content_bytes.assert_called_once()
        assert content == b"audio_data"

    def test_get_content_summary(self, mock_media_file):
        """Test getting content summary."""
        artifact = AudioArtifact(
            name="Test Audio",
            media_file=mock_media_file,
            duration_seconds=120.5
        )

        summary = artifact.get_content_summary()

        mock_media_file.get_content_summary.assert_called_once()
        assert "Audio:" in summary
        assert "(120.5s)" in summary

    def test_get_content_summary_no_duration(self, mock_media_file):
        """Test getting content summary without duration."""
        artifact = AudioArtifact(
            name="Test Audio",
            media_file=mock_media_file
        )

        summary = artifact.get_content_summary()

        assert "Audio:" in summary
        assert "s)" not in summary  # No duration suffix

    def test_get_size_bytes(self, mock_media_file):
        """Test getting size in bytes."""
        artifact = AudioArtifact(
            name="Test Audio",
            media_file=mock_media_file
        )

        size = artifact.get_size_bytes()

        assert size == 1024000

    def test_is_accessible(self, mock_media_file):
        """Test accessibility check."""
        artifact = AudioArtifact(
            name="Test Audio",
            media_file=mock_media_file
        )

        accessible = artifact.is_accessible()

        mock_media_file.is_accessible.assert_called_once()
        assert accessible is True

    def test_get_mime_type(self, mock_media_file):
        """Test getting MIME type."""
        artifact = AudioArtifact(
            name="Test Audio",
            media_file=mock_media_file
        )

        mime_type = artifact.get_mime_type()

        assert mime_type == "audio/mpeg"

    def test_get_file_extension(self, mock_media_file):
        """Test getting file extension."""
        artifact = AudioArtifact(
            name="Test Audio",
            media_file=mock_media_file
        )

        extension = artifact.get_file_extension()

        assert extension == ".mp3"

    def test_get_file_extension_no_extension(self, mock_media_file):
        """Test getting file extension when name has no extension."""
        mock_media_file.name = "audio_file_no_extension"

        artifact = AudioArtifact(
            name="Test Audio",
            media_file=mock_media_file
        )

        extension = artifact.get_file_extension()

        assert extension is None

    def test_get_file_extension_no_name(self, mock_media_file):
        """Test getting file extension when media file has no name."""
        mock_media_file.name = None

        artifact = AudioArtifact(
            name="Test Audio",
            media_file=mock_media_file
        )

        extension = artifact.get_file_extension()

        assert extension is None