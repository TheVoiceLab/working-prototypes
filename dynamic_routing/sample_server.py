from typing import List

from fastapi import FastAPI
from pydantic import BaseModel

from dynamic_routing import build_graph, load_skills
import uvicorn

app = FastAPI(title="Skill Builder API")

# build graph once when server starts
skills_data = load_skills("sample-skill-builder.csv")
graph = build_graph(skills_data)


class RequestModel(BaseModel):
    user_input: str
    user_instruction: str


class ResponseModel(BaseModel):
    outputs: List[str]
    errors: List[str]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate", response_model=ResponseModel)
async def generate(req: RequestModel):

    state = {
        "user_input": req.user_input,
        "user_instruction": req.user_instruction,
        "outputs": [],
        "errors": []
    }
    print(f"user_input: {req.user_input}")

    result = graph.invoke(state)

    return ResponseModel(
        outputs=result.get("outputs", []),
        errors=result.get("errors", [])
    )

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",   # allow external access
        port=6600,        # change to any port you want
        reload=False       # auto reload for development
    )