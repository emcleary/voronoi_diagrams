from typing import Protocol, Self


class SupportLT(Protocol):
    def __lt__(self, value: Self, /) -> bool: ...
