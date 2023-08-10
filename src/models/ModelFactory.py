import enum

from src.models import GNNModels


class Models(enum.Enum):
    """
    Supported models
    """
    shape_gnn = 1


def create_model(model_type: Models, num_classes, in_features, **kwargs):
    # create the specified model
    if model_type is Models.shape_gnn:
        model = GNNModels.ShapeGNN(in_features, num_classes, **kwargs)
    else:
        raise ValueError(f'model type {model_type} unknown')

    return model
