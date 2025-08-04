import pytest
import tempfile
import yaml
from pathlib import Path
from box import ConfigBox
from src.textSummariser.utils.common import read_yaml, create_directories


class TestCommonUtils:
    """Test common utility functions"""

    def test_read_yaml_valid_file(self, temp_dir):
        """Test reading a valid YAML file"""
        # Create a test YAML file
        test_data = {
            "model": {"name": "pegasus", "epochs": 5},
            "data": {"batch_size": 16},
        }

        yaml_file = temp_dir / "test_config.yaml"
        with open(yaml_file, "w") as f:
            yaml.dump(test_data, f)

        # Test reading the file
        result = read_yaml(yaml_file)

        assert isinstance(result, ConfigBox)
        assert result.model.name == "pegasus"
        assert result.model.epochs == 5
        assert result.data.batch_size == 16

    def test_read_yaml_empty_file(self, temp_dir):
        """Test reading an empty YAML file"""
        yaml_file = temp_dir / "empty.yaml"
        yaml_file.touch()  # Create empty file

        with pytest.raises(ValueError, match="yaml file is empty"):
            read_yaml(yaml_file)

    def test_read_yaml_nonexistent_file(self, temp_dir):
        """Test reading a non-existent YAML file"""
        yaml_file = temp_dir / "nonexistent.yaml"

        with pytest.raises(FileNotFoundError):
            read_yaml(yaml_file)

    def test_create_directories_single(self, temp_dir):
        """Test creating a single directory"""
        new_dir = temp_dir / "new_directory"

        create_directories([str(new_dir)], verbose=False)

        assert new_dir.exists()
        assert new_dir.is_dir()

    def test_create_directories_multiple(self, temp_dir):
        """Test creating multiple directories"""
        dirs = [
            temp_dir / "dir1",
            temp_dir / "dir2" / "subdir",
            temp_dir / "dir3",
        ]

        create_directories([str(d) for d in dirs], verbose=False)

        for directory in dirs:
            assert directory.exists()
            assert directory.is_dir()

    def test_create_directories_existing(self, temp_dir):
        """Test creating directories that already exist"""
        existing_dir = temp_dir / "existing"
        existing_dir.mkdir()

        # Should not raise an error
        create_directories([str(existing_dir)], verbose=False)

        assert existing_dir.exists()
        assert existing_dir.is_dir()
