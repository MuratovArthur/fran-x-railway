import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from transformers import pipeline as hf_pipeline_ner
from dotenv import load_dotenv
import torch
from typing import List, Dict, Any

load_dotenv()

app = FastAPI(title="FRaN-X Inference Server")

# Load once at startup
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN not set in .env")

# Use the same model IDs as Streamlit (can be overridden by env vars)
CLS_MODEL_ID = os.getenv("CLS_MODEL", "artur-muratov/franx-cls")
NER_MODEL_ID = os.getenv("NER_MODEL", "artur-muratov/franx-ner")

# Stage 2 classification - matching Streamlit implementation
cls_tokenizer = AutoTokenizer.from_pretrained(CLS_MODEL_ID, token=HF_TOKEN)
cls_model = AutoModelForSequenceClassification.from_pretrained(
    CLS_MODEL_ID, token=HF_TOKEN, torch_dtype=torch.float16, device_map="auto"
)
cls_pipeline = pipeline(
    "text-classification",
    model=cls_model,
    tokenizer=cls_tokenizer,
    top_k=None,  # return all scores like Streamlit
    device_map="auto"
)

# Stage 1 NER - using standard pipeline (Note: This may not match custom DebertaV3NerClassifier exactly)
ner_pipeline = hf_pipeline_ner(
    "token-classification",
    model=NER_MODEL_ID,
    aggregation_strategy="simple",
    token=HF_TOKEN,
    device=0 if torch.cuda.is_available() else -1
)

class TextIn(BaseModel):
    text: str

class ClassifyIn(BaseModel):
    entity_mention: str
    p_main_role: str
    context: str
    threshold: float = 0.01
    margin: float = 0.05

class NERSpan(BaseModel):
    start: int
    end: int
    entity_text: str
    role: str
    prob_antagonist: float
    prob_protagonist: float
    prob_innocent: float
    prob_unknown: float

@app.post("/ner")
def ner(in_data: TextIn) -> List[Dict[str, Any]]:
    """
    NER endpoint that tries to match the Streamlit DebertaV3NerClassifier output format.
    Note: This uses standard HF pipeline which may not exactly match the custom implementation.
    """
    try:
        # Get token classification results
        raw_ents = ner_pipeline(in_data.text)
        
        # Convert to format expected by Streamlit
        processed_spans = []
        for ent in raw_ents:
            # Convert numpy types to Python types
            start = int(ent['start'])
            end = int(ent['end'])
            score = float(ent['score'])
            label = ent['entity_group'] if 'entity_group' in ent else ent['entity']
            
            # Ensure start <= end
            if start > end:
                start, end = end, start
            
            # Extract entity text 
            entity_text = in_data.text[start:end]
            
            # Adjust boundaries to remove leading/trailing whitespace (correct logic)
            stripped_text = entity_text.strip()
            if stripped_text:
                # Find where the stripped text starts and ends within the original entity
                left_spaces = len(entity_text) - len(entity_text.lstrip())
                right_spaces = len(entity_text) - len(entity_text.rstrip())
                
                clean_start = start + left_spaces
                clean_end = end - right_spaces  # Fixed calculation
                clean_entity_text = stripped_text
                
                # Validation - ensure we have a valid range
                if clean_start >= clean_end or clean_start < 0 or clean_end > len(in_data.text):
                    print(f"WARNING: Invalid span - start:{start}, end:{end}, clean_start:{clean_start}, clean_end:{clean_end}, entity_text:'{entity_text}', label:{label}")
                    continue
                    
                # Double-check the entity text extraction
                extracted_text = in_data.text[clean_start:clean_end]
                if extracted_text != clean_entity_text:
                    print(f"WARNING: Text mismatch - expected:'{clean_entity_text}', extracted:'{extracted_text}'")
                    # Use the extracted text as ground truth
                    clean_entity_text = extracted_text
            else:
                # If entity is all whitespace, skip it
                continue
            
            # Map label to role probabilities (this is approximate - the real model has specific probabilities)
            role_probs = {
                'prob_antagonist': 0.0,
                'prob_protagonist': 0.0, 
                'prob_innocent': 0.0,
                'prob_unknown': 0.0
            }
            
            # This is a rough mapping - the actual model would have specific probability outputs
            if 'ANTAGONIST' in label.upper():
                role_probs['prob_antagonist'] = score
                role = 'Antagonist'
            elif 'PROTAGONIST' in label.upper():
                role_probs['prob_protagonist'] = score
                role = 'Protagonist'
            elif 'INNOCENT' in label.upper():
                role_probs['prob_innocent'] = score
                role = 'Innocent'
            else:
                role_probs['prob_unknown'] = score
                role = 'Unknown'
            
            processed_spans.append({
                'start': clean_start,
                'end': clean_end,
                'entity_text': clean_entity_text,
                'role': role,
                **role_probs
            })
        
        return processed_spans
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/classify")
def classify(in_data: ClassifyIn) -> Dict[str, Any]:
    """
    Classification endpoint that matches Streamlit's stage2 format
    """
    try:
        # Format input text exactly like Streamlit does
        input_text = (
            f"Entity: {in_data.entity_mention}\n"
            f"Main Role: {in_data.p_main_role}\n"
            f"Context: {in_data.context}"
        )
        
        # Get classification scores
        raw_scores = cls_pipeline(input_text)[0]
        
        # Convert numpy types and filter by threshold
        scores = {}
        for score_item in raw_scores:
            label = score_item['label']
            score_val = float(score_item['score'])
            if score_val > in_data.threshold:
                scores[label] = round(score_val, 4)
        
        if not scores:
            return {
                'predicted_fine_with_scores': {},
                'predicted_fine_margin': [],
                'p_fine_roles_w_conf': {}
            }
        
        # Select roles within margin of top score (like Streamlit does)
        top_score = max(scores.values())
        margin_roles = [role for role, score in scores.items() 
                       if score >= top_score - in_data.margin]
        
        # Filter scores to only include margin roles
        filtered_scores = {role: scores[role] for role in margin_roles}
        
        return {
            'predicted_fine_with_scores': scores,
            'predicted_fine_margin': margin_roles,
            'p_fine_roles_w_conf': filtered_scores
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
def health():
    return {"status": "healthy", "models_loaded": True}

# Debug endpoint to see raw NER output
@app.post("/debug-ner")
def debug_ner(in_data: TextIn) -> Dict[str, Any]:
    """Debug endpoint to see what the raw NER pipeline returns"""
    try:
        raw_ents = ner_pipeline(in_data.text)
        return {
            "text": in_data.text,
            "text_length": len(in_data.text),
            "raw_entities": raw_ents,
            "entity_count": len(raw_ents)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
