from pydantic import BaseModel, Field
from typing import List

class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., description="Sepal length (cm)")
    sepal_width: float = Field(..., description="Sepal width (cm)")
    petal_length: float = Field(..., description="Petal length (cm)")
    petal_width: float = Field(..., description="Petal width (cm)")

class PredictRequest(BaseModel):
    records: List[IrisFeatures]

class PredictResponse(BaseModel):
    predictions: List[str]
    probabilities: List[list]
