from pydantic import BaseModel


class Alert(BaseModel):
    """Represents a cleaned crisis alert."""

    titre: str
    pays: str
    severite: str



# if __name__ == "__main__":
#     alerte = Alert(
#         titre=123,
#         pays="germany",
#         severite="orange"
#     )
#     print(alerte)

