import nbformat as nbf
import os

def add_save_cell_to_lstm(path):
    with open(path, 'r', encoding='utf-8') as f:
        nb = nbf.read(f, as_version=4)
    
    save_code = """# ==========================================
# 💾 13. Save Model
# ==========================================
import os
os.makedirs('../models', exist_ok=True)
model.save('../models/lstm_forecast_v1.h5')
print("✅ Model saved to models/lstm_forecast_v1.h5")"""
    
    # Check if last cell is already save or if we can just append a cell
    new_cell = nbf.v4.new_code_cell(save_code)
    nb.cells.append(new_cell)
    
    with open(path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"Added save cell to {path}")

def add_save_cell_to_bertopic(path):
    with open(path, 'r', encoding='utf-8') as f:
        nb = nbf.read(f, as_version=4)
        
    save_code = """# 5. Save Model
import os
os.makedirs('../models', exist_ok=True)
topic_model.save("../models/bertopic_model")
print("✅ BERTopic model saved to models/bertopic_model")"""
    
    new_cell = nbf.v4.new_code_cell(save_code)
    nb.cells.append(new_cell)
    
    with open(path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"Added save cell to {path}")

lstm_path = 'notebook-Phase-2/Forecasting_LSTM (1).ipynb'
bertopic_path = 'notebook-Phase-2/Model_BERTopic.ipynb'

if os.path.exists(lstm_path):
    add_save_cell_to_lstm(lstm_path)
if os.path.exists(bertopic_path):
    add_save_cell_to_bertopic(bertopic_path)
