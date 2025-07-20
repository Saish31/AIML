# Pneumonia Detection from Chest X-Rays using Transfer Learning and XAI

This project develops an end-to-end deep learning pipeline to classify chest X-ray images for pneumonia and implements Explainable AI (XAI) to interpret the model's decisions. The final model is deployed in an interactive web application built with Streamlit.

---

### 📸 Dashboard Screenshot
<img width="274" height="293" alt="image" src="https://github.com/user-attachments/assets/5d82c029-bf80-4409-a98c-c6637141db31" />

---

### ✨ Key Features

* **Advanced Deep Learning:** Utilizes **Transfer Learning** with a pre-trained VGG16 model to achieve high performance on a specialized medical dataset.
* **Class Imbalance Handling:** Implements **Data Augmentation** and **Class Weights** to effectively train on a highly imbalanced dataset.
* **Smart Training:** Uses **Early Stopping** and **Learning Rate Scheduling** callbacks to prevent overfitting and automatically find the optimal model.
* **Explainable AI (XAI):** Implements **Grad-CAM** to produce heatmaps that visualize the regions of an X-ray the model uses for its predictions, adding trust and interpretability.
* **Rigorous Evaluation:** Assesses model performance using a **Confusion Matrix**, **Precision**, **Recall**, and **F1-Score** to understand its clinical relevance.
* **Interactive Dashboard:** Deploys the final model into a user-friendly Streamlit application that allows for real-time image uploads and analysis.

---

### 🔬 Final Model Performance

The final model was evaluated on an unseen test set of 624 images. The results show a model that is highly sensitive to detecting pneumonia, which is a desirable trait for a medical screening tool.

**Classification Report:**
<img width="446" height="299" alt="image" src="https://github.com/user-attachments/assets/188283a5-ee18-4db9-8e35-21f97bd7063c" />

The model achieved an excellent **98% recall for pneumonia**, indicating it is very unlikely to miss a positive case. This came at the cost of a lower recall (61%) for normal cases, a common and often acceptable trade-off in medical diagnostic models.  

---  
  
### 🛠️ Tech Stack  
  
* **Language:** Python  
* **Libraries:** TensorFlow/Keras, Scikit-learn, OpenCV, Streamlit, Matplotlib, NumPy, Pillow

---

The dataset used is https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia  
