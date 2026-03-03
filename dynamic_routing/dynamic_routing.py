import csv
import os
import random
import operator
from typing import TypedDict, Dict, List, Any, Annotated

from langgraph.graph import StateGraph, END
from langgraph.types import Send

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# -----------------------------
# API key
# -----------------------------
api_key = os.getenv("openai-key")
if not api_key:
    raise EnvironmentError("OPENAI_API_KEY not set in environment")
api_key = api_key.strip()

model_name = "gpt-4o-mini"


# -----------------------------
# State definition
# -----------------------------
class SkillBuilderState(TypedDict, total=False):
    user_instruction: str
    user_input: str
    generated_story: str
    selected_skills: List[str]
    # IMPORTANT: Use Annotated + operator.add so parallel nodes
    # APPEND to these lists instead of overwriting each other.
    outputs: Annotated[List[str], operator.add]
    errors: Annotated[List[str], operator.add]


# -----------------------------
# Load skill builder CSV
# -----------------------------
def load_skills(csv_file: str) -> Dict[str, Dict[str, Any]]:
    skills: Dict[str, Dict[str, Any]] = {}
    # Fallback for demonstration if file is missing
    if not os.path.exists(csv_file):
        return {
            "vocabulary_builder": {"description": "Focuses on nouns and verbs", "max-level": 5},
            "pronoun_builder": {"description": "Focuses on I, you, he, she", "max-level": 5}
        }
    with open(csv_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            skills[row["name"]] = {
                "description": row["description"],
                "max_level": int(row["max-level"]),
            }
    return skills


# -----------------------------
# Skill node factory
# -----------------------------
def make_skill_node(name: str, description: str):
    def node(state: SkillBuilderState) -> Dict[str, Any]:
        # RESTORED: The print statement you needed to see activation
        print(f'skill {name} has been activated.......................')

        instruction = state["user_instruction"]
        story = state["generated_story"]

        roll = random.random()
        if roll < 0.10:  # Increased slightly for demo purposes
            return {"errors": [f"{name}: soft error"], "outputs": []}
        else:
            output = (
                f"[{name}] Generated 1 page worksheet based on a story of "
                f"{len(story)} letters following instruction: {instruction}."
            )
            return {"outputs": [output], "errors": []}

    return node


# -----------------------------
# Story generator node
# -----------------------------
def story_generator_node(state: SkillBuilderState) -> Dict[str, Any]:
    llm = ChatOpenAI(model=model_name, api_key=api_key)
    prompt = f"Write a short, engaging children's story about: '{state['user_input']}'"
    response = llm.invoke([HumanMessage(content=prompt)])
    print("=== Generated story ===")
    print(f"{response.content.strip()[:300]}...")
    return {"generated_story": response.content}


# -----------------------------
# Skill selector node
# -----------------------------
def skill_selector_node(state: SkillBuilderState, skills: Dict[str, Dict[str, Any]]):
    llm = ChatOpenAI(model=model_name, api_key=api_key)
    skill_descriptions = "\n".join(f"- {name}: {info['description']}" for name, info in skills.items())

    prompt = f"""
The user instruction is: "{state['user_instruction']}"
Available skill builders:
{skill_descriptions}
Return ONLY the list of selected skill builder names in the format like [ "vocabulary_builder", "prediction_builder"]

"""
    response = llm.invoke([HumanMessage(content=prompt)])
    raw = response.content.strip()

    print("=== skill selector response ===")
    print(raw)

    try:
        # Using literal_eval for safety over eval
        import ast
        selected = ast.literal_eval(raw)
        selected_skills = [s for s in selected if s in skills]
    except Exception:
        selected_skills = []

    return {"selected_skills": selected_skills}


# -----------------------------
# Merge node (fan-in)
# -----------------------------
def merge_node(state: SkillBuilderState) -> Dict[str, Any]:
    print("=== All selected skills have finished. Merging. ===")
    return {}


# -----------------------------
# Build LangGraph
# -----------------------------
def build_graph(skills: Dict[str, Dict[str, Any]]):
    builder = StateGraph(SkillBuilderState)

    builder.add_node("story_generator", story_generator_node)
    builder.add_node("skill_selector", lambda s: skill_selector_node(s, skills))
    builder.add_node("merge", merge_node)

    for skill_name, info in skills.items():
        builder.add_node(skill_name, make_skill_node(skill_name, info["description"]))

    builder.set_entry_point("story_generator")
    builder.add_edge("story_generator", "skill_selector")

    # -----------------------------
    # DYNAMIC FAN-OUT using Send
    # -----------------------------
    def route_from_selector(state: SkillBuilderState):
        selected = state.get("selected_skills", [])
        print(f"========= skills selected by LLM: {selected} =========")

        if not selected:
            return "merge"

        # Send() creates a parallel branch for each selected skill
        return [Send(skill, state) for skill in selected]

    # Map the dynamic Send strings to the actual nodes
    builder.add_conditional_edges(
        "skill_selector",
        route_from_selector,
        {**{name: name for name in skills.keys()}, "merge": "merge"}
    )

    # -----------------------------
    # FAN-IN: Connect skills back to merge
    # -----------------------------
    for skill_name in skills.keys():
        builder.add_edge(skill_name, "merge")

    builder.add_edge("merge", END)

    return builder.compile()


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    skills_data = load_skills("sample-skill-builder.csv")
    graph = build_graph(skills_data)

    #----------The first query---------------
    initial_state = {
        "user_input": "Leon like to kick the seat on the schoolbus...",
        "user_instruction": "My son understands the words but does not understand the plots",
        "outputs": [],
        "errors": []
    }

    result = graph.invoke(initial_state)

    print("\n" + "=" * 40)
    print("FINAL OUTPUTS:")
    for out in result.get("outputs", []):
        print(f"✅ {out}")

    if result.get("errors"):
        print("\nERRORS ENCOUNTERED:")
        for err in result.get("errors", []):
            print(f"❌ {err}")
    print("=" * 40)

    # ----------The second query---------------
    initial_state = {
        "user_input": "Ressie is Leon's friend",
        "user_instruction": "My son understands addition or substraction in real situration",
        "outputs": [],
        "errors": []
    }

    result = graph.invoke(initial_state)

    print("\n" + "=" * 40)
    print("FINAL OUTPUTS:")
    for out in result.get("outputs", []):
        print(f"✅ {out}")

    if result.get("errors"):
        print("\nERRORS ENCOUNTERED:")
        for err in result.get("errors", []):
            print(f"❌ {err}")
    print("=" * 40)