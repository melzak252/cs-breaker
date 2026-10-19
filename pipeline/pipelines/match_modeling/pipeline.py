"""Pipeline definition for match modeling and player rankings."""
from kedro.pipeline import Pipeline, node
from .nodes import (
    generate_player_rankings_and_features,
    train_baseline_prediction_models,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=generate_player_rankings_and_features,
                inputs="consolidated_matches",
                outputs=["match_features_with_ratings", "player_rankings_master",
                         "modeling_data_audit", "player_rating_state"],
                name="generate_rankings_and_features_node",
            ),
            node(
                func=train_baseline_prediction_models,
                inputs="match_features_with_ratings",
                outputs=["model_evaluation_metrics", "baseline_test_predictions", "baseline_model"],
                name="train_baseline_prediction_models_node",
            ),
        ]
    )
