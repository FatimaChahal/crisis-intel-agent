import json
import os


class S3Client:
    """
    Local simulation of S3 storage.
    Saves and loads files from local filesystem instead of AWS S3.
    """

    def __init__(self, base_path: str = "data"):
        """
        Initialize the local S3 client.

        Args:
            base_path: Base directory to simulate S3 bucket.
        """
        self.base_path = base_path

    def upload(self, key: str, data: list) -> None:
        """
        Save data to a local file (simulates S3 upload).

        Args:
            key: File path relative to base_path (e.g. 'bronze/alerts.json').
            data: List of dictionaries to save.
        """
        path = os.path.join(self.base_path, key)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"✅ Uploaded to {path}")

    def download(self, key: str) -> list:
        """
        Load data from a local file (simulates S3 download).

        Args:
            key: File path relative to base_path.

        Returns:
            List of dictionaries loaded from file.
        """
        path = os.path.join(self.base_path, key)
        with open(path, "r") as f:
            return json.load(f)


if __name__ == "__main__":
    client = S3Client()

    # Simuler un upload vers Bronze
    alerts = [
        {"titre": "  FLOOD IN GERMANY  ", "pays": "  germany  ", "severite": "  ORANGE  "},
        {"titre": "  EARTHQUAKE IN TURKEY  ", "pays": "  turkey  ", "severite": "  RED  "},
    ]
    client.upload("bronze/alerts_s3.json", alerts)

    # Simuler un download depuis Bronze
    data = client.download("bronze/alerts_s3.json")
    print(f"Downloaded {len(data)} alerts")
    print(data[0])