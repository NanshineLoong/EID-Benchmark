"""Base dataset class and data item container."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator


@dataclass
class DataItem:
    """Container for a single evaluation item.

    Attributes:
        case_id: Unique identifier for the case
        task: Case description/prompt for the model
        answer: Ground truth answer/diagnosis
        patient_facts: List of patient facts (for simulators)
        exam_facts: List of examination/test facts (for simulators)
        raw: Original raw data from dataset file
    """

    case_id: str
    task: str
    answer: str
    patient_facts: list[str] = field(default_factory=list)
    exam_facts: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


class Dataset:
    """Container for evaluation dataset.

    Provides iteration over DataItem instances and basic metadata.
    """

    def __init__(
        self,
        items: list[DataItem],
        name: str = "unknown",
        max_items: int | None = None,
    ) -> None:
        """Initialize dataset.

        Args:
            items: List of DataItem instances
            name: Dataset name
            max_items: Maximum number of items to include
        """
        self.name = name
        self._items = items[:max_items] if max_items else items

    def __len__(self) -> int:
        """Return number of items in dataset."""
        return len(self._items)

    def __iter__(self) -> Iterator[DataItem]:
        """Iterate over dataset items."""
        return iter(self._items)

    def __getitem__(self, index: int) -> DataItem:
        """Get item by index."""
        return self._items[index]

    def get_by_id(self, case_id: str) -> DataItem | None:
        """Get item by case ID.

        Args:
            case_id: Case identifier to look up

        Returns:
            DataItem if found, None otherwise
        """
        for item in self._items:
            if item.case_id == case_id:
                return item
        return None
