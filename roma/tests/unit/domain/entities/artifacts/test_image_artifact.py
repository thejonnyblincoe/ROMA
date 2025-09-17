"""
Tests for ImageArtifact following Agno media patterns.

Tests cover all content sources (URL, file, bytes), validation,
serialization, and Agno-compatible methods.
"""

import base64
import pytest
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

from src.roma.domain.entities.artifacts.image_artifact import ImageArtifact
from src.roma.domain.value_objects.media_type import MediaType


class TestImageArtifact:
    """Test ImageArtifact following Agno patterns."""
    
    def test_content_source_validation_success(self):
        """Test valid single content source initialization."""
        # Test with bytes content
        image_bytes = ImageArtifact(
            name="test_image",
            content=b"fake_image_data",
            mime_type="image/png"
        )
        assert image_bytes.content == b"fake_image_data"
        assert image_bytes.url is None
        assert image_bytes.filepath is None
        
        # Test with URL
        image_url = ImageArtifact(
            name="test_image",
            url="https://example.com/image.png",
            mime_type="image/png"
        )
        assert image_url.url == "https://example.com/image.png"
        assert image_url.content is None
        assert image_url.filepath is None
        
        # Test with filepath
        image_file = ImageArtifact(
            name="test_image",
            filepath="/path/to/image.png",
            mime_type="image/png"
        )
        assert image_file.filepath == Path("/path/to/image.png")
        assert image_file.content is None
        assert image_file.url is None
    
    def test_content_source_validation_failure(self):
        """Test validation fails with multiple or no content sources."""
        # Test with no content sources
        with pytest.raises(ValueError, match="Exactly one content source"):
            ImageArtifact(name="test_image", mime_type="image/png")
        
        # Test with multiple content sources
        with pytest.raises(ValueError, match="Exactly one content source"):
            ImageArtifact(
                name="test_image",
                content=b"data",
                url="https://example.com/image.png",
                mime_type="image/png"
            )
    
    def test_mime_type_validation(self):
        """Test MIME type validation."""
        # Valid MIME types
        valid_types = [
            "image/png", "image/jpeg", "image/gif", "image/webp",
            "image/bmp", "image/tiff", "image/svg+xml"
        ]
        
        for mime_type in valid_types:
            image = ImageArtifact(
                name="test", 
                content=b"data", 
                mime_type=mime_type
            )
            assert image.mime_type == mime_type
        
        # Invalid MIME type
        with pytest.raises(ValueError, match="Invalid image MIME type"):
            ImageArtifact(
                name="test",
                content=b"data",
                mime_type="text/plain"
            )
    
    def test_filepath_conversion(self):
        """Test filepath string to Path conversion."""
        image = ImageArtifact(
            name="test",
            filepath="/path/to/image.png",
            mime_type="image/png"
        )
        assert isinstance(image.filepath, Path)
        assert str(image.filepath) == "/path/to/image.png"
    
    def test_media_type_property(self):
        """Test media_type property returns IMAGE."""
        image = ImageArtifact(
            name="test",
            content=b"data",
            mime_type="image/png"
        )
        assert image.media_type == MediaType.IMAGE
    
    @pytest.mark.asyncio
    async def test_get_content_from_bytes(self):
        """Test getting content from bytes."""
        content = b"fake_image_data"
        image = ImageArtifact(
            name="test",
            content=content,
            mime_type="image/png"
        )
        
        result = await image.get_content()
        assert result == content
    
    @pytest.mark.asyncio
    async def test_get_content_from_file(self):
        """Test getting content from file."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            test_content = b"fake_image_data"
            tmp.write(test_content)
            tmp.flush()
            
            image = ImageArtifact(
                name="test",
                filepath=tmp.name,
                mime_type="image/png"
            )
            
            result = await image.get_content()
            assert result == test_content
            
            # Cleanup
            Path(tmp.name).unlink()
    
    @pytest.mark.asyncio
    async def test_get_content_from_url(self):
        """Test getting content from URL."""
        mock_response = AsyncMock()
        mock_response.content = b"fake_image_data"
        mock_response.headers = {"content-type": "image/png"}
        mock_response.raise_for_status = AsyncMock()
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_client_instance = AsyncMock()
            mock_client_instance.get.return_value = mock_response
            mock_client.return_value.__aenter__.return_value = mock_client_instance
            
            image = ImageArtifact(
                name="test",
                url="https://example.com/image.png",
                mime_type="image/png"
            )
            
            result = await image.get_content()
            assert result == b"fake_image_data"
    
    def test_get_content_summary(self):
        """Test content summary generation."""
        image = ImageArtifact(
            name="test_image",
            content=b"data",
            mime_type="image/png",
            format="PNG",
            width=100,
            height=200
        )
        
        summary = image.get_content_summary()
        assert "Image: test_image" in summary
        assert "format=PNG" in summary
        assert "size=100x200" in summary
        assert "source=bytes" in summary
    
    def test_get_size_bytes_from_content(self):
        """Test size calculation from bytes content."""
        content = b"fake_image_data"
        image = ImageArtifact(
            name="test",
            content=content,
            mime_type="image/png"
        )
        
        assert image.get_size_bytes() == len(content)
    
    def test_get_size_bytes_from_file(self):
        """Test size calculation from file."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            test_content = b"fake_image_data"
            tmp.write(test_content)
            tmp.flush()
            
            image = ImageArtifact(
                name="test",
                filepath=tmp.name,
                mime_type="image/png"
            )
            
            assert image.get_size_bytes() == len(test_content)
            
            # Cleanup
            Path(tmp.name).unlink()
    
    def test_is_accessible(self):
        """Test accessibility checks for different content sources."""
        # Bytes content - always accessible
        image_bytes = ImageArtifact(
            name="test",
            content=b"data",
            mime_type="image/png"
        )
        assert image_bytes.is_accessible() is True
        
        # URL - assumed accessible
        image_url = ImageArtifact(
            name="test",
            url="https://example.com/image.png",
            mime_type="image/png"
        )
        assert image_url.is_accessible() is True
        
        # Non-existent file - not accessible
        image_file = ImageArtifact(
            name="test",
            filepath="/non/existent/path.png",
            mime_type="image/png"
        )
        assert image_file.is_accessible() is False
    
    def test_get_mime_type(self):
        """Test MIME type getter."""
        image = ImageArtifact(
            name="test",
            content=b"data",
            mime_type="image/jpeg"
        )
        assert image.get_mime_type() == "image/jpeg"
    
    def test_get_file_extension_from_format(self):
        """Test file extension detection from format."""
        test_cases = [
            ("PNG", ".png"),
            ("JPEG", ".jpg"),
            ("GIF", ".gif"),
            ("WEBP", ".webp"),
        ]
        
        for format_name, expected_ext in test_cases:
            image = ImageArtifact(
                name="test",
                content=b"data",
                mime_type="image/png",
                format=format_name
            )
            assert image.get_file_extension() == expected_ext
    
    def test_get_file_extension_from_mime_type(self):
        """Test file extension detection from MIME type."""
        test_cases = [
            ("image/png", ".png"),
            ("image/jpeg", ".jpg"),
            ("image/gif", ".gif"),
            ("image/webp", ".webp"),
        ]
        
        for mime_type, expected_ext in test_cases:
            image = ImageArtifact(
                name="test",
                content=b"data",
                mime_type=mime_type
            )
            assert image.get_file_extension() == expected_ext
    
    @pytest.mark.asyncio
    async def test_to_base64_agno_pattern(self):
        """Test base64 encoding following Agno pattern."""
        content = b"fake_image_data"
        image = ImageArtifact(
            name="test",
            content=content,
            mime_type="image/png"
        )
        
        # Without data URL
        b64 = await image.to_base64(include_data_url=False)
        expected_b64 = base64.b64encode(content).decode('utf-8')
        assert b64 == expected_b64
        
        # With data URL
        data_url = await image.to_base64(include_data_url=True)
        assert data_url == f"data:image/png;base64,{expected_b64}"
    
    def test_from_base64_agno_pattern(self):
        """Test creation from base64 following Agno pattern."""
        content = b"fake_image_data"
        b64_str = base64.b64encode(content).decode('utf-8')
        
        # Simple base64
        image = ImageArtifact.from_base64(
            base64_str=b64_str,
            name="test_image",
            mime_type="image/png"
        )
        
        assert image.name == "test_image"
        assert image.content == content
        assert image.mime_type == "image/png"
        
        # Data URL format
        data_url = f"data:image/jpeg;base64,{b64_str}"
        image_data_url = ImageArtifact.from_base64(
            base64_str=data_url,
            name="test_image"
        )
        
        assert image_data_url.content == content
        assert image_data_url.mime_type == "image/jpeg"
    
    def test_from_url_agno_pattern(self):
        """Test creation from URL following Agno pattern."""
        url = "https://example.com/image.png"
        image = ImageArtifact.from_url(
            url=url,
            name="test_image",
            mime_type="image/png"
        )
        
        assert image.url == url
        assert image.name == "test_image"
        assert image.mime_type == "image/png"
        assert image.content is None
        assert image.filepath is None
    
    def test_from_file_agno_pattern(self):
        """Test creation from file following Agno pattern."""
        filepath = "/path/to/image.jpg"
        
        # With explicit name and MIME type
        image = ImageArtifact.from_file(
            filepath=filepath,
            name="custom_name",
            mime_type="image/jpeg"
        )
        
        assert image.filepath == Path(filepath)
        assert image.name == "custom_name"
        assert image.mime_type == "image/jpeg"
        
        # Auto-detected name and MIME type
        image_auto = ImageArtifact.from_file(filepath="/path/to/test.png")
        assert image_auto.name == "test.png"
        assert image_auto.mime_type == "image/png"
    
    def test_to_dict_agno_pattern(self):
        """Test dictionary serialization following Agno pattern."""
        image = ImageArtifact(
            name="test_image",
            content=b"fake_data",
            mime_type="image/png",
            format="PNG",
            width=100,
            height=200
        )
        
        # Without content
        dict_data = image.to_dict(include_content=False)
        
        expected_fields = [
            "artifact_id", "name", "media_type", "created_at",
            "url", "filepath", "format", "width", "height"
        ]
        
        for field in expected_fields:
            assert field in dict_data
        
        assert "content_base64" not in dict_data
        assert dict_data["media_type"] == "image"
        assert dict_data["name"] == "test_image"
        
        # With content
        dict_with_content = image.to_dict(include_content=True)
        assert "content_base64" in dict_with_content
        expected_b64 = base64.b64encode(b"fake_data").decode('utf-8')
        assert dict_with_content["content_base64"] == expected_b64
    
    def test_from_dict_agno_pattern(self):
        """Test creation from dictionary following Agno pattern."""
        content = b"fake_data"
        content_b64 = base64.b64encode(content).decode('utf-8')
        
        dict_data = {
            "name": "test_image",
            "content_base64": content_b64,
            "mime_type": "image/png",
            "format": "PNG",
            "width": 100,
            "height": 200,
            "filepath": "/path/to/image.png"
        }
        
        image = ImageArtifact.from_dict(dict_data)
        
        assert image.name == "test_image"
        assert image.content == content
        assert image.mime_type == "image/png"
        assert image.format == "PNG"
        assert image.width == 100
        assert image.height == 200
        # filepath is cleared when content_base64 is provided (only one content source allowed)
        assert image.filepath is None
    
    def test_immutability(self):
        """Test that ImageArtifact is immutable."""
        image = ImageArtifact(
            name="test",
            content=b"data",
            mime_type="image/png"
        )
        
        # Should not be able to modify fields
        with pytest.raises(Exception):  # ValidationError or AttributeError
            image.name = "modified"
        
        with pytest.raises(Exception):
            image.content = b"modified"
    
    def test_string_representation(self):
        """Test string representation."""
        image = ImageArtifact(
            name="test_image",
            content=b"data",
            mime_type="image/png",
            task_id="task_123"
        )
        
        str_repr = str(image)
        assert "ImageArtifact" in str_repr
        assert "test_image" in str_repr
        assert "image" in str_repr
        assert "task_123" in str_repr


class TestImageArtifactIntegration:
    """Integration tests for ImageArtifact with real files."""
    
    def test_real_file_integration(self):
        """Test with real temporary image file."""
        # Create a minimal PNG file (1x1 pixel)
        png_data = bytes([
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,  # PNG signature
            0x00, 0x00, 0x00, 0x0D,  # IHDR chunk length
            0x49, 0x48, 0x44, 0x52,  # IHDR
            0x00, 0x00, 0x00, 0x01,  # Width: 1
            0x00, 0x00, 0x00, 0x01,  # Height: 1
            0x08, 0x02, 0x00, 0x00, 0x00,  # Bit depth, color type, etc.
            0x90, 0x77, 0x53, 0xDE,  # CRC
            0x00, 0x00, 0x00, 0x0C,  # IDAT chunk length
            0x49, 0x44, 0x41, 0x54,  # IDAT
            0x08, 0x99, 0x01, 0x01, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0x00, 0x00, 0x00,
            0x02, 0x00, 0x01,  # IDAT data + CRC
            0x00, 0x00, 0x00, 0x00,  # IEND chunk length
            0x49, 0x45, 0x4E, 0x44,  # IEND
            0xAE, 0x42, 0x60, 0x82   # CRC
        ])
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp.write(png_data)
            tmp.flush()
            
            # Test from_file
            image = ImageArtifact.from_file(tmp.name)
            
            assert image.name == Path(tmp.name).name
            assert image.mime_type == "image/png"
            assert image.is_accessible()
            assert image.get_size_bytes() == len(png_data)
            
            # Cleanup
            Path(tmp.name).unlink()
    
    @pytest.mark.asyncio
    async def test_base64_round_trip(self):
        """Test round-trip base64 encoding/decoding."""
        original_content = b"test_image_data_12345"
        
        # Create from content
        image1 = ImageArtifact(
            name="test",
            content=original_content,
            mime_type="image/png"
        )
        
        # Convert to base64
        b64_str = await image1.to_base64()
        
        # Create from base64
        image2 = ImageArtifact.from_base64(
            base64_str=b64_str,
            name="test",
            mime_type="image/png"
        )
        
        # Verify content is preserved
        assert image2.content == original_content
        assert image2.name == "test"
        assert image2.mime_type == "image/png"