"""
test_gemini_client.py
Simulates API outputs using mocks to test the Gemini_client wrappers 
without making real network requests.
"""

import types
import numpy as np
from unittest.mock import patch, MagicMock
from RAG.Pipeline import Gemini_client as gc

# Force client mock so initialization doesn't try to read real keys/env
gc.client = MagicMock()

def test_embed_normalized():
    print("\n[TEST] embed_normalized -> converts API structures and normalizes vectors")
    
    # Mocking the return structure from embed_content
    fake_embedding_1 = types.SimpleNamespace(values=[3.0, 4.0])  # norm of [3, 4] is 5
    fake_embedding_2 = types.SimpleNamespace(values=[0.0, 5.0])  # norm is 5
    fake_api_response = types.SimpleNamespace(embeddings=[fake_embedding_1, fake_embedding_2])

    with patch("RAG.Pipeline.Gemini_client.call_function_with_handling", return_value=fake_api_response) as mock_caller:
        texts = ["text1", "text2"]
        vecs = gc.embed_normalized(texts)
        
        # Check that the caller was used with the right arguments
        mock_caller.assert_called_once()
        assert mock_caller.call_args[1]["contents"] == texts
        assert mock_caller.call_args[1]["model"] == "gemini-embedding-001"
        
        # Verify normalization: lengths of vectors must be exactly 1.0
        assert np.allclose(np.linalg.norm(vecs, axis=1), [1.0, 1.0])
        # [3/5, 4/5] -> [0.6, 0.8] and [0/5, 5/5] -> [0.0, 1.0]
        assert np.allclose(vecs[0], [0.6, 0.8])
        assert np.allclose(vecs[1], [0.0, 1.0])
        print("PASS")


def test_embed():
    print("\n[TEST] embed -> forwards request to standard embedding model")
    fake_api_response = types.SimpleNamespace(embeddings=[types.SimpleNamespace(values=[1, 2, 3])])
    
    with patch("RAG.Pipeline.Gemini_client.call_function_with_handling", return_value=fake_api_response) as mock_caller:
        texts = ["hello world"]
        result = gc.embed(texts)
        
        mock_caller.assert_called_once()
        assert mock_caller.call_args[1]["model"] == "gemini-embedding-2"
        assert result.embeddings[0].values == [1, 2, 3]
        print("PASS")


def test_generate():
    print("\n[TEST] generate -> handles LLM content generation")
    fake_response = types.SimpleNamespace(text="My mocked answer.")
    
    with patch("RAG.Pipeline.Gemini_client.call_function_with_handling", return_value=fake_response) as mock_caller:
        prompt = "Explain quantum physics."
        config = {"temperature": 0.5}
        result = gc.generate(prompt, config=config)
        
        mock_caller.assert_called_once()
        assert mock_caller.call_args[1]["model"] == "gemini-2.5-flash"
        assert mock_caller.call_args[1]["contents"] == prompt
        assert mock_caller.call_args[1]["config"] == config
        assert result.text == "My mocked answer."
        print("PASS")


if __name__ == "__main__":
    test_embed_normalized()
    test_embed()
    test_generate()
    print("\nAll Gemini client tests passed.")