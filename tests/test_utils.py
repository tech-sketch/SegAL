from segal.utils import load_json, save_json


def test_save_json(fixture_path) -> None:
    """Test save_json function"""
    # Arrange
    data = [1, 2, 4]
    save_path = fixture_path / "data.json"

    # Act
    save_json(data, save_path)

    # Assert
    assert save_path.exists() is True
    save_path.unlink()


def test_load_json(fixture_path) -> None:
    """Test load_json function"""
    # Arrange
    expected = [1, 2, 4]
    load_path = fixture_path / "test_data.json"

    # Act
    data = load_json(load_path)

    # Assert
    assert data == expected
