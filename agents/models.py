from pydantic import BaseModel, Field


class CoderResult(BaseModel):
    code: str = Field(..., description="The generated Python (or relevant) code.")
    explanation: str = Field(..., description="Natural language explanation of the solution.")
    plan: str = Field(..., description="Execution plan / reasoning steps used to solve the task.")
    result: str = Field(..., description="Result of running the code, including stdout/stderr.")
