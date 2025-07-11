# %%
import logging
import sys
import os
import re
from typing import Dict, Any, Literal, Optional, TypeVar, Generic
from ast import literal_eval
from pydantic import BaseModel, ValidationError
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph
from src.embedding.vector_store_manager import VectorStoreManager

# Ensure the project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# %%
# Define a TypeVar to allow generic types in FieldWithJustification
T = TypeVar('T')

# New class to encapsulate value and justification
class FieldWithJustification(BaseModel, Generic[T]):
    value: T
    justification: str

class PaperMetadata(BaseModel):
    PUBLICATION_DATE: FieldWithJustification[Optional[str]] # Optional because it can be NULL
    CODE_AVAILABLE: FieldWithJustification[Literal["Yes", "No"]]
    DATA_AVAILABLE: FieldWithJustification[Literal["Yes", "No"]]
    MULTIMODALITY: FieldWithJustification[Literal["no", "data", "semantic", "data_semantic"]]
    POPULATION: FieldWithJustification[Literal["Yes", "No"]]
    INTERVENTION: FieldWithJustification[Literal["Yes", "No"]]
    CONTEXT: FieldWithJustification[Literal["Yes", "No"]]
    OUTCOME: FieldWithJustification[Literal["Yes", "No"]]

class ArticleMetadataAgent:
    """
    Agent specialized in extracting metadata for the PAPER table.
    """
    def __init__(self, vector_store_manager: VectorStoreManager, llm_model: Any):
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        self.vector_store_manager = vector_store_manager
        self.llm_model = llm_model
        self.logger.info("Initialized ArticleMetadataAgent")

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes metadata extraction for a given article.
        Args:
            state (dict): A state dictionary containing:
            - article_id (str): Article identifier
            - article_full_text (str): Complete article text fallback
        Returns:
            dict: Updated state with validated metadata under 'paper_metadata'.
        """
        article_id = state.get("article_id")
        article_full_text = state.get("article_full_text", "")
        self.logger.info(f"Starting metadata extraction for article ID: {article_id}")

        # Step 1: RAG Context Retrieval
        query_text = """
        Information on the **article's publication date** (e.g., copyright year, accepted, published).
        Details regarding the **availability of source code** (e.g., GitHub, repository, implementation) and **research data** (e.g., dataset link, public data, data shared).
        Description of the **input data characteristics and multimodality** (e.g., combining numeric, text, image data, or distinct financial contexts like stocks and cryptocurrencies).
        Mentions of **specific machine learning interventions** used (e.g., Graph Neural Networks (GNN), attention mechanisms, Transformer, Recurrent Neural Networks (RNN), LSTM, GRU, semi-supervised learning).
        The **study's application context in financial decision-making** (e.g., trading, investment, risk management) and the **prediction of financial market outcomes** (e.g., price forecasting, volatility estimation, market movement).
        """
        results = self.vector_store_manager.query_vector_store(
            query_text=query_text,
            n_results=30,
            filter_metadata={"article_id": article_id}
        )
        context_chunks = [r["document"] for r in results if r.get("document")]
        context_for_llm = "\n".join(context_chunks) if context_chunks else article_full_text

        # Step 2: Prompt
        prompt_template = f"""
        You are a data extraction agent specialized in scientific article metadata.
        Your task is to extract the following fields and return them as a Python dictionary.
        For each field, provide the **extracted value** and a **concise justification** explaining where and how the value was derived from the article, referencing specific sections or sentences.
        Follow the validation rules for each field. If a field is not explicitly mentioned, use the default value according to the instructions below and justify the default.

        Article Text: ```{context_for_llm}```
        
        ---
        **Field Definitions and Extraction Rules:**

        *   **`PUBLICATION_DATE`** (YYYY-MM-DD or NULL):
            *   Extract the publication date from the article's metadata (e.g., copyright notice, header, footer).
            *   If only the year is available (e.g., "© 2020 Springer"), assume the date is the last day of that year (e.g., "2020-12-31").
            *   If no date is found, use NULL.
            *   If a date was identified, justify your answer by reproducing the corresponding page number and the text segment from the article where the date appears.
        
        *   **`CODE_AVAILABLE`** ("Yes" or "No"):
            *   Set to "Yes" if the article **explicitly mentions** available source code (e.g., provides a GitHub link, mentions a public repository, or states "code available upon request").
            *   Otherwise, set to "No".
            *   If "Yes", justify your answer by reproducing the relevant page number and the text segment from the article where this information appears.
        
        *   **`DATA_AVAILABLE`** ("Yes" or "No"):
            *   Set to "Yes" if the article **explicitly mentions** accessible data (e.g., provides a link to a data repository, mentions a public dataset, or states "data shared").
            *   Otherwise, set to "No".
            *   If "Yes", justify your answer by reproducing the relevant page number and the text segment from the article where this information appears.
            
        *   **`MULTIMODALITY`** ("no" | "data" | "semantic" | "data_semantic"):
            *   Determine based on the input data **explicitly stated as integrated into a single model**:
                *   "**no**": Inputs are from a single context with no significant semantic or data-type variation (e.g., OHLCV data for one stock), or diverse data are analyzed by separate models without integration.
                *   "**data**": Inputs to a single model include fundamentally **different data types** (e.g., numeric prices + text news, text + audio, categorical + numeric + video, graph + numeric + image), regardless of domain.
                *   "**semantic**": Inputs to a single model are of the **same data type** but originate from **distinct contexts or domains with meaningfully different semantics or dynamics** (e.g., stock market OHLCV + cryptocurrency OHLCV, oil prices + gold prices, bonds + equities). The distinction must reflect separate operational frameworks or market behaviors, not just different attributes within one context.
                *   "**data_semantic**": A combination of "data" and "semantic" in a single model (e.g., numeric stock prices + numeric crypto prices + text news).
            *   **Evaluation Checklist for MULTIMODALITY**:
                1.  **Confirm Data Types**: Explicitly identify all input data types (e.g., numeric, text, image) used in the model. If only one data type is present, "data" and "data_semantic" are not applicable.
                2.  **Verify Integration**: Check if the paper explicitly states that inputs from multiple contexts or types are fed into a **single model instance**. If inputs are processed by separate models or evaluated independently, classify as "no" unless integration is clear.
                3.  **Assess Context Distinction**: For "semantic," ensure contexts (e.g., stocks vs. cryptocurrencies) have distinct operational frameworks or dynamics, not just different assets or attributes within the same market.
                4.  **Default to "no" on Ambiguity**: If the paper does not clearly specify integration of diverse data types or contexts into a single model, assume "no" multimodalidade.
            *   **Justification**:
                1. Justify the classification based on input data types and contexts, and whether they are integrated into a *single model*.
                2. Justify the chosen classification by summarizing the evidence from the article that supports your decision.
        
        *   **`POPULATION`** ("Yes" or "No"):
            *   Set to "Yes" if the `MULTIMODALITY` field (determined above) is "data", "semantic", or "data_semantic".
            *   Otherwise, set to "No".
            *   Justify based on the MULTIMODALITY rule.
        
        *   **`INTERVENTION`** ("Yes" or "No"):
            *   Set to "Yes" if the proposed model **explicitly uses** any of the following techniques: **Graph Neural Networks (GNN)**, **attention-based models** (e.g., Transformer), **Recurrent Neural Networks (RNN**, including LSTM, GRU), or **semi-supervised learning**.
            *   Otherwise, set to "No".
            *   Justify by mentioning the specific techniques.
            
        *   **`CONTEXT`** ("Yes" or "No"):
            *   Set to "Yes" if the study **explicitly focuses** on **financial decision-making tasks** (e.g., trading, investment, risk management, portfolio optimization).
            *   Otherwise, set to "No".
            *   Justify the answer by stating the type(s) of context identified and quoting the exact page number and text excerpt from the article where this information is found.
            
        *   **`OUTCOME`** ("Yes" or "No"):
            *   Set to "Yes" if the model **explicitly predicts financial market outcomes** (e.g., price forecasting, volatility estimation, market movement prediction, portfolio returns).
            *   Otherwise, set to "No".
            *   Justify the answer by stating the type(s) of prediction performed and quoting the exact page and text excerpt from the article where this information is mentioned.
        
        Output Format (Python Dictionary):

        ```python
        {{
            "PUBLICATION_DATE": {{"value": "YYYY-MM-DD or NULL", "justification": "Detailed explanation"}},
            "CODE_AVAILABLE": {{"value": "Yes" or "No", "justification": "Detailed explanation"}},
            "DATA_AVAILABLE": {{"value": "Yes" or "No", "justification": "Detailed explanation"}},
            "MULTIMODALITY": {{"value": "no" | "data" | "semantic" | "data_semantic", "justification": "Detailed explanation"}},
            "POPULATION": {{"value": "Yes" or "No", "justification": "Detailed explanation"}},
            "INTERVENTION": {{"value": "Yes" or "No", "justification": "Detailed explanation"}},
            "CONTEXT": {{"value": "Yes" or "No", "justification": "Detailed explanation"}},
            "OUTCOME": {{"value": "Yes" or "No", "justification": "Detailed explanation"}}
        }}
        ```
        """
        try:
            llm_response = self.llm_model.invoke([HumanMessage(content=prompt_template)])
            llm_response_str = llm_response.content.strip()

            # Extract Python dict from code block using regex
            match = re.search(r"```python\s*(\{.*?\})\s*```", llm_response_str, re.DOTALL)
            extracted_str = match.group(1) if match else llm_response_str
            parsed_dict = literal_eval(extracted_str)

            validated = PaperMetadata(**parsed_dict)
            self.logger.info(f"Successfully validated metadata for {article_id}.")

            # validated.dict() will already return the structure with 'value' and 'justification'
            return {**state, "paper_metadata": validated.dict()}

        except (ValidationError, Exception) as e:
            self.logger.error(f"Metadata extraction failed for {article_id}: {e}")
            return {**state, "paper_metadata": {}}


# Example: minimal LangGraph integration (as node)
def build_metadata_graph(agent: ArticleMetadataAgent):
    builder = StateGraph()
    builder.add_node("extract_metadata", agent.run)
    builder.set_entry_point("extract_metadata")
    return builder.compile()

# def build_metadata_graph(agent: ArticleMetadataAgent):
#     """ Builds a simple LangGraph representing the metadata extraction process. 
#         This function could be expanded to include more nodes and complex logic.
#     """
#     builder = StateGraph(Dict[str, Any]) # Define the state type for clarity
#     builder.add_node("extract_metadata", agent.run) # Add the agent's run method as a node
#     builder.set_entry_point("extract_metadata") # Set this node as the entry point
#     return builder.compile() # Compile the graph for execution


# Test runner
if __name__ == "__main__":
    from langchain_openai import ChatOpenAI

    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')

    vector_store_manager = VectorStoreManager()
    llm = ChatOpenAI(temperature=0, model="gpt-4")

    agent = ArticleMetadataAgent(vector_store_manager=vector_store_manager, llm_model=llm)

    sample_state = {
        "article_id": "CI9TCB38",
        "article_full_text": """
        This paper introduces a novel CNN-LSTM model for predicting gold prices.
        The authors explicitly provide their code on GitHub and the dataset used for training is available on Kaggle.
        The input data consists solely of historical gold prices (numeric time series).
        The model's architecture incorporates an LSTM layer. This study focuses on enhancing financial forecasting
        and the model predicts future gold market movements.
        """
    }

    result_state = agent.run(sample_state)
    print("\nExtracted PAPER Metadata:")
    print(result_state.get("paper_metadata"))

# %%
