"""Pipeline definition for data consolidation."""
from kedro.pipeline import Pipeline, node
from .nodes import build_consolidated_dataset


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=build_consolidated_dataset,
                inputs=["hltv_matches_raw", "kaggle_matches_raw", "scraped_lineups_processed",
                        "historical_maps_raw", "historical_players_raw"],
                outputs="consolidated_matches",
                name="consolidate_all_matches_node",
            ),
        ]
    )
