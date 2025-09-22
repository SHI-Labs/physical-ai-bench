from typing import Any, Dict, Optional
from uuid import UUID

import attrs

@attrs.define
class MetricScore:
    """
    A class representing a score for a metric.

    Attributes:
        metric_name (str): The name of the metric.
        metric_subcategory_value (Optional[Dict[str, float]]): A dictionary representing subcategories of the metric
            with corresponding values.
            For example, the Vbench metric has 16 dimensions such as subject_consistency, background_consistency, etc.
            Each entry in `metric_subcategory_value` consists of the subcategory name and its corresponding values.
            metric_value (Optional[float]): The score or value of the metric itself. This field is required if
            `metric_subcategory_value` is not provided.
        config (Optional[Dict[str, Any]]): Additional configuration or metadata related to the metric score.
        timestamp (Optional[str]): The timestamp when the metric score was generated.
        run_uuid (Optional[UUID]): The UUID of the run when the metric score was generated.
    """

    metric_name: str
    metric_subcategory_value: Optional[Dict[str, float]] = None
    metric_value: Optional[float] = None
    config: Optional[Dict[str, Any]] = None
    timestamp: Optional[str] = None
    run_uuid: Optional[UUID] = None

    def __post_init__(self) -> None:
        if not self.metric_subcategory_value and not self.metric_value:
            raise ValueError("Both 'metric_subcategory_value' and 'metric_value' cannot be empty.")


@attrs.define
class MetricContainer:
    metric_name: str
    metric_value: Optional[float] = None
