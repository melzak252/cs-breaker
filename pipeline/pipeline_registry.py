"""Project pipeline registry."""
from typing import Dict
from kedro.pipeline import Pipeline
from pipeline.pipelines.data_consolidation import pipeline as consolidation_pipeline
from pipeline.pipelines.match_modeling import pipeline as match_modeling_pipeline


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    consolidation = consolidation_pipeline.create_pipeline()
    modeling = match_modeling_pipeline.create_pipeline()

    return {
        "__default__": consolidation + modeling,
        "consolidation": consolidation,
        "modeling": modeling,
    }
