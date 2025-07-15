**# Transformer-based English–Hindi Translator**  

**A PyTorch implementation of a Transformer model built from scratch to translate English sentences to Hindi.**  

---

**## Features**  

* **Custom multi-head self-attention** and **cross-attention** modules  
* **Sinusoidal positional encoding** with caching for efficiency  
* **Character-level tokenization** (can be adapted to subword tokenization)  
* **Training with teacher forcing**, **padding masking**, and **mixed-precision support**  

---

**## Requirements**

* **Python 3.8+**
* **PyTorch 1.8+**
* **torchtext** (for dataset utilities)
* **numpy**

*Optional (for faster training):*

* NVIDIA GPU with CUDA
* Google Colab / Cloud GPU runtime

---


**## Usage**  

1. **Prepare parallel corpus files:**  

   * `train.en`: English sentences (one per line)  
   * `train.hi`: Corresponding Hindi sentences (one per line)
   * These files were too big to upload on github. You can get them from here: https://www.kaggle.com/datasets/mathurinache/samanantar  
2. **Adjust hyperparameters** in `Trainer.ipynb` (batch size, learning rate, epochs).  
3. **Run training** in a Jupyter/Colab environment.  
---

**## Training**  

* **Default:** 10 epochs, batch size 32  
* **Uses mixed-precision** (AMP) if CUDA is available  

---  

**## Evaluation / Inference**  

Within `Trainer.ipynb`, an evaluation snippet generates translations using greedy decoding:  

```python  
# Example: translate "should we go to the mall?"  
eng_sentence = ("should we go to the mall?",)  
hi_sentence = ("",)  
# Loop up to max length, appending predicted tokens  
```  

**## License**  

**This project is released under the MIT License. See `LICENSE` for details.**  
