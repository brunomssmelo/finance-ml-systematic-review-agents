# %%
# === Imports
import logging
from src.embedding.vector_store_manager import  VectorStoreManager
from src.agents.article_metadata_agent import ArticleMetadataAgent

# %%
from langchain_openai import ChatOpenAI

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')

vector_store_manager = VectorStoreManager()
# llm = ChatOpenAI(temperature=0, model="gpt-4o")
llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
# llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

agent = ArticleMetadataAgent(vector_store_manager=vector_store_manager, llm_model=llm)


# %%
# sample_state = {
#     "article_id": "CI9TCB38",
#     "article_full_text": """
#     This paper introduces a novel CNN-LSTM model for predicting gold prices.
#     The authors explicitly provide their code on GitHub and the dataset used for training is available on Kaggle.
#     The input data consists solely of historical gold prices (numeric time series).
#     The model's architecture incorporates an LSTM layer. This study focuses on enhancing financial forecasting
#     and the model predicts future gold market movements.
#     """
# }

sample_state = {
    "article_id": "8W5LXMJZ",
    "article_full_text": """Article not found."""
}

result_state = agent.run(sample_state)
print("\nExtracted PAPER Metadata:")
print(result_state.get("paper_metadata"))

# %%
