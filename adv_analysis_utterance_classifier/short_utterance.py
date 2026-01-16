import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from typing import List, Dict
import json

# ---------------- CONFIG ---------------- #
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "gpt-4o-mini"

TOP_K = 5
SIM_THRESHOLD = 0.35
MARGIN_THRESHOLD = 0.05
LLM_CONF_THRESHOLD = 0.6
SHORT_UTTER_SIM_THRESHOLD = 0.7  # Fuzzy short-utterance similarity, very high confidence level
#if not that high, we still give it to LLM
SHORT_TOKEN_CUTOFF = 3           # ≤3 tokens considered short
# --------------------------------------- #
class IntentDefinition:
    def __init__(self, index: int | None = None, row=None):
        if index is None or row is None:
            # NO MATCH intent
            self.intent_id = "NO_MATCH"
            self.intent_name = "no match"
            self.intent_desc = "no match"
            self.examples = []
            self.embed_text = ""
            self.similarity = 0.0
        else:
            self.intent_id = f"Intent_{index}"
            self.intent_name = row["Name"]
            self.intent_desc = row["Description"]
            self.examples = [
                e.lower().strip().strip('"').strip("'")
                for e in str(row["Sample Utterances"]).split("\n") if e.strip()
            ]
            self.embed_text = (
                f"Name: {row['Name']}. "
                f"Description: {row['Description']}. "
                f"Examples: {', '.join(self.examples)}"
            )
            self.similarity = -0.1

class ClassificationResult:
    def __init__(self, intent:IntentDefinition, confidence:float, method:str):
        self.intent = intent
        self.confidence = confidence
        self.method = method

class IntentRAGClassifier:
    def __init__(self, excel_path: str, openai_key: str):
        # --- OpenAI client ---
        self.client = OpenAI(api_key=openai_key)
        # --- Embedding model ---
        self.model = SentenceTransformer(EMBED_MODEL)
        # --- Load intents ---
        self.intent_df = pd.read_csv(excel_path)
        self.intent_records = self._build_intent_records()

        self.intent_id_map = {}
        for intent in self.intent_records:
            self.intent_id_map[intent.intent_id] = intent

        # --- Compute embeddings for FAISS ---
        self.intent_embeddings = self.model.encode(
            [r.embed_text for r in self.intent_records],
            normalize_embeddings=True
        ).astype("float32")

        # --- FAISS index ---
        dim = self.intent_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # cosine similarity
        self.index.add(self.intent_embeddings)
        print(f"Loaded {len(self.intent_records)} intents into FAISS")

        # --- Short utterance map ---
        self.short_utterance_map = self._build_short_utterance_map()
        self._build_short_utterance_embeddings()

    # ---------- DATA PREP ---------- #
    def _build_intent_records(self) -> List[IntentDefinition]:
        records = []
        for index, row in self.intent_df.iterrows():
            records.append(IntentDefinition(int(index), row))
        return records

    def _build_short_utterance_map(self) -> Dict[str, Dict]:
        """Build dictionary for short utterances (≤2 words)"""
        mapping = {}
        for rec in self.intent_records:
            for ex in rec.examples:
                if len(ex.split()) <= 2:
                    mapping[ex] = {
                        "intent_id": rec.intent_id,
                        "intent_name": rec.intent_name,
                    }
        return mapping

    def _build_short_utterance_embeddings(self):
        """Compute embeddings for fuzzy short-utterance matching"""
        self.short_utterance_keys = list(self.short_utterance_map.keys())
        if self.short_utterance_keys:
            self.short_utterance_embeddings = self.model.encode(
                self.short_utterance_keys,
                normalize_embeddings=True
            ).astype("float32")
        else:
            self.short_utterance_embeddings = np.array([])

    # ---------- CLASSIFY ---------- #
    def classify(self, utterance: str) -> ClassificationResult:
        utter_lower = utterance.lower().strip()
        token_count = len(utter_lower.split())

        short_u_intent = None
        best_sim = -1
        # ---------- Short-utterance matcher ----------
        if token_count <= SHORT_TOKEN_CUTOFF and len(self.short_utterance_keys) > 0:
            utter_emb = self.model.encode([utter_lower], normalize_embeddings=True).astype("float32")
            sims = np.dot(self.short_utterance_embeddings, utter_emb.T).flatten()
            best_idx = sims.argmax()
            best_sim = sims[best_idx]
            key = self.short_utterance_keys[best_idx]
            intent = self.short_utterance_map[key]
            short_u_intent = intent
            if best_sim >= SHORT_UTTER_SIM_THRESHOLD:
                return  ClassificationResult(
                    intent=self.intent_id_map[short_u_intent["intent_id"]], confidence=best_sim, method='short utterance vec matching'
                )

        # ---------- FAISS retrieval for longer utterances ----------
        utter_emb = self.model.encode([utterance], normalize_embeddings=True).astype("float32")
        sims, idxs = self.index.search(utter_emb, TOP_K)
        sims = sims[0]
        idxs = idxs[0]

        candidates = []
        for pos, i in enumerate(idxs):
            the_candidate = self.intent_records[i]
            the_candidate.similarity = float(sims[pos])
            candidates.append(the_candidate)

        if short_u_intent is not None:
            short_intent = self.intent_id_map[short_u_intent["intent_id"]]
            candidates.append(short_intent)

        # ---------- Semantic gating ----------
        failed, reason = self._semantic_gate(candidates)
        if failed:
            return self._unknown(reason, candidates)

        # ---------- LLM disambiguation ----------
        llm_result = self._llm_decide(utterance, candidates)

        print(f'===========Parsed LLM Result============\n{llm_result}')

        if llm_result["intent_id"] == "NONE":
            return self._unknown("llm_reject", candidates)
        if llm_result["confidence"] < LLM_CONF_THRESHOLD:
            return self._unknown("low_llm_confidence", candidates)

        return ClassificationResult(
            intent = self.intent_id_map[llm_result["intent_id"]],
            confidence = llm_result["confidence"],
            method =  "faiss+rag+llm"
        )


    # ---------- SEMANTIC GATING ---------- #
    def _semantic_gate(self, candidates):
        if candidates[0].similarity < SIM_THRESHOLD and candidates[-1].similarity < SIM_THRESHOLD:
            return True, "low_similarity"

        if len(candidates) > 1:
            margin = candidates[0].similarity - candidates[1].similarity
            if margin < MARGIN_THRESHOLD:
                return False, "ambiguous_match"
        return False, None

    # ---------- LLM DECISION ---------- #
    def _llm_decide(self, utterance: str, candidates: List[IntentDefinition]) -> Dict:
        options = "\n".join(
            f"{i+1}. {c.intent_id}: {c.intent_name}, {c.intent_desc}, (examples: {', '.join(c.examples)})"
            for i, c in enumerate(candidates)
        )
        prompt = f"""
You are an intent classifier.

Utterance:`
"{utterance}"

Candidate intents:
{options}

Rules:
- Choose exactly ONE intent_id from the list
- If none match, answer NONE
- Return confidence from 0.0–1.0

Respond in JSON only:
{{"intent_id": "...", "confidence": 0.0}}
"""
        def parse_llm_json(text: str) -> dict:
            text = text.strip()
            print(f"====>{text}")

            try:
                return json.loads(text)
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse JSON from LLM response: {text}") from e

        print(f'=========LLM Prompt==========\n{prompt}')

        resp = self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        parsed = parse_llm_json(resp.choices[0].message.content)
        print(f'=========LLM Response==========\n{parsed}')
        if parsed["intent_id"] == "NONE":
            return {"intent_id": "NONE", "confidence": 0.0}

        intent = self.intent_id_map[parsed["intent_id"]]
        intent.similarity = float(parsed["confidence"])

        if intent is None:
            return {
                "intent_id": "UNKNOWN",
                "reason": "llm_invalid_intent",
                "llm_output": parsed
            }

        return {
            "intent_id": intent.intent_id,
            "intent_name": intent.intent_name,
            "confidence": intent.similarity,
        }

    # ---------- UNKNOWN ---------- #
    def _unknown(self, reason, candidates):
        """
        Return ClassificationResult for unknown intents.
        Confidence is estimated from candidates if available, else 0.
        """
        # Default confidence
        conf = 0.0

        # Use top candidate similarity if candidates exist
        if candidates:
            conf = max(c.similarity for c in candidates)

        return ClassificationResult(
            intent=IntentDefinition(),
            confidence=conf,
            method=f"unknown ({reason})"
        )


# ---------------- MAIN ---------------- #
if __name__ == "__main__":
    # --- Load API key ---
    cert_file = r"C:\Users\minru\Documents\certs.txt"
    with open(cert_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    api_key = data["openai_key"]

    # --- Initialize classifier ---
    classifier = IntentRAGClassifier(
        excel_path=r"C:\faa-data\intents.csv",
        openai_key=api_key,
    )

    # --- Test utterances ---
    tests = [
        "some random stuff",
        "program this remote",
        "I don't have internet",
        "sure",
        "no way",
        "talk to a customer service",
        "I have to do everything stay on the phone make a call I don't have to tell you what to do you're the 1 who you",
        "re on the 15 years going for 8 to 12 hours a day make a phone call",
        "technical issue",
        "getting a error code rdk d1000 on all my TVs",
        "payment arrangement",
        "payment extension",
        "necesito hablar con un representante",
        "questions about my bill",
        "no picture on TV",
        "what a representative about the phone my phone service",
        "disconnect service",
        "operator",
        "Wi - Fi issues",
        "yep",
        "account question at 796",
        "missing channels on my lineup",
        "I need to speak to someone I have a problem that you can't fix I need to talk to someone it's regarding my bill",
        "um hello",
        "return equipment",
        "Wi-Fi connections",
        "TV won't start",
        "I need to find out the balance on what we owe on our phone that we're purchasing through you",
        "speak to",
        "outage",
        "my television",
        "um not getting any pictures on my TV is coming on but all this doing is saying unable to connect you can check your devices internet connection",
    ]

    for t in tests:
        result = classifier.classify(t)
        print(result.intent.intent_name)
        print(result.confidence)
        print(result.method)
        break