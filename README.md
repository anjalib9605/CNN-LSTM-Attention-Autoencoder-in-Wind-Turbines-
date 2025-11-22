# Neural Network Based Fault Detection in Wind Turbines using CNN-LSTM-Attention model
This project focuses on building a CNN-LSTM-Attention deep learning model for fault detection in Wind Turbine data collected from CARE to Compare dataset in Kaggle. The model is designed as an autoencoder that learns normal system behavior and identifies anomalies based on reconstruction error. Demonstrates a complete end-to-end workflow including data preprocessing, sequence generation, model design, training, and evaluation.


##  Key Features
-Multivariate time-series anomaly detection
-Hybrid model using CNN (local feature extraction) + LSTM (temporal modeling) + Attention (important timestep weighting)
-Autoencoder reconstruction framework
-Data preprocessing and feature scaling
-Sliding-window sequence creation
-Reconstruction error–based thresholding
-Visualization of anomaly scores 

---

##  Model Architecture
The architecture is designed to capture both short-range patterns through CNN and long-range dependencies with LSTM, enhanced with an attention layer to identify the most influential timesteps in the sequence.
The model integrates three powerful components:

### **1. Convolutional Neural Networks (1D CNN)**
- Extract local temporal patterns  
- Reduce noise  
- Capture short-term dependencies  

### **2. LSTM Encoder-Decoder**
- Model long-range temporal relationships  
- Compress sequence into latent representation  
- Reconstruct original signal  

### **3. Attention Layer**
- Assigns importance weights to time steps  
- Helps the model focus on critical temporal regions  

This combination allows robust modeling of both local and global patterns in time-series data.

---


## Project Workflow

### **1. Data Loading**
- Import input CSV files 
- Merge into a single unified dataframe  
- Basic structure checks & validation  

### **2. Preprocessing**
- Handle missing values  
- Scale features using `StandardScaler`  
- Remove duplicates or unwanted fields  
- Ensure consistent feature ordering  

### **3. Sequence Construction**
Convert raw time-series into fixed-length sliding windows.

### **4. Model Training**
-Train only using normal behavioral data
-Use early stopping
-Loss function: Mean Squared Error (MSE)
-Store trained weights

### **5. Anomaly Detection**
-Compute reconstruction error on test sequences
-Higher error = higher anomaly score
-Select threshold using validation data or statistical distribution

### **Results Summary**
-The autoencoder effectively learns normal system behavior
-Anomalous data exhibits significantly higher reconstruction errors
-Attention layer highlights critical regions contributing to anomalies
-AUC (anomaly score): 0.7790873539917227

---

# **Technologies Used**
-Python <br>
-NumPy <br>
-Pandas <br>
-TensorFlow / Keras <br>
-Scikit-learn <br>
-Matplotlib <br>
-Jupyter Notebook 


## Project Collaborators

This work is part of a collaborative research effort.  
Contributors include:

- **Anjali B**   
- **Sreevidya PA**   
- **Noel Sushan**  

All contributors participated in the design, development, and analysis of the model.
