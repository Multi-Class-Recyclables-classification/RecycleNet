# RecycleNet: Multi-Class Recyclables Classification

RecycleNet is a deep learning project focused on the multi-class classification of recyclable and non-recyclable waste materials from images. The goal is to develop an accurate and robust model capable of distinguishing between various waste categories to aid in automated sorting and recycling efforts.

---

## 1. Problem Definition and Data Collection

The core problem addressed by RecycleNet is the accurate classification of waste images into distinct categories. This is a crucial step in automating waste management systems, which can significantly improve recycling efficiency and reduce contamination.

The project utilizes a custom-collected dataset, as evidenced by the repository structure and team member tasks. The dataset is structured to represent a multi-class classification challenge, with the following seven categories identified through initial data exploration:

| Category | Description |
| :--- | :--- |
| **biological** | Organic waste |
| **cardboard** | Cardboard materials |
| **glass** | Glass materials |
| **metal** | Metal materials |
| **paper** | Paper materials |
| **plastic** | Plastic materials |
| **trash** | General non-recyclable waste |

Team members Amr Reffat and Zaid Ahmed were actively involved in the data collection process, ensuring a relevant and representative dataset for the classification task.

---

## 2. Data Cleaning and Analysis

Initial data analysis was performed using the `dataExploration.ipynb` notebook to understand the dataset's characteristics, including class distribution.

The analysis revealed a slight class imbalance, particularly in the **glass** category, which had a lower image count compared to the other six categories. This finding informed the subsequent model training strategy, likely necessitating techniques to mitigate the effects of class imbalance.

---

## 3. Feature Engineering

For image classification tasks, feature engineering primarily involves image preprocessing and augmentation techniques applied to the raw pixel data. The project leverages the PyTorch ecosystem, specifically `torchvision.transforms`, for this purpose.

Standard feature engineering steps applied to the image data likely included:
*   **Resizing and Cropping:** Normalizing image dimensions for model input.
*   **Normalization:** Scaling pixel values to a standard range (e.g., mean and standard deviation normalization) as required by pre-trained models like ResNet and ViT.
*   **Data Augmentation:** Techniques such as random rotations, flips, and color jittering were likely used to artificially expand the dataset and improve model generalization.

---

## 4. Model Design

The RecycleNet project adopted a multi-pronged approach to model development, exploring various state-of-the-art architectures to achieve optimal performance. The models implemented include:

| Model Type | Architecture | Team Member(s) |
| :--- | :--- | :--- |
| **Vision Transformer (ViT)** | State-of-the-art transformer model for vision tasks. | Amr Reffat, Zaid Ahmed |
| **Fine-tuned ResNet** | A pre-trained Residual Network (ResNet) fine-tuned for the 7-class problem. | Seif Eldeen Nasser |
| **Custom CNN** | A Convolutional Neural Network designed from scratch or a simple base. | Omar Shohieb, Mohamed Osama |

This comparative approach ensures a robust solution by testing the efficacy of both classic CNNs and modern transformer-based models on the specific waste classification challenge.

---

## 5. Model Training

Model training was conducted using the **PyTorch** deep learning framework, as indicated by the imports in the project notebooks and utility files.

The training process involved:
*   **Data Loading:** Utilizing PyTorch's `DataLoader` and `ImageFolder` for efficient batch processing.
*   **Optimization:** Employing standard optimizers (e.g., Adam or SGD) and a suitable loss function (likely Cross-Entropy Loss for multi-class classification).
*   **Iterative Refinement:** Multiple model checkpoints and versions (`best_model`, `best_model_3`, `model_checkpoints`, etc.) suggest an iterative training and hyperparameter tuning process to achieve the best possible accuracy.

---

## 6. Model Testing and Inference

Model testing and inference are handled through a dedicated pipeline. The models are evaluated using standard classification metrics from `sklearn.metrics`, including the **Confusion Matrix** and **Classification Report**, to assess performance across all seven classes.

For deployment, the inference logic is encapsulated in a `predict` function, which is integrated into the application's backend. This function handles the loading of the best-performing model checkpoint and processes new image data for real-time classification.

---

## 7. GUI Implementation and Application Running

The RecycleNet application is implemented as a high-performance **RESTful API** using the **FastAPI** framework, rather than a traditional graphical user interface (GUI). This design choice allows for flexible integration with various front-end applications or other services.

The core functionality is exposed via the `/classify` endpoint in `main.py`, which performs the following steps:
1.  Accepts an image file upload.
2.  Validates the image file size (up to 10 MB) and format (PNG, JPG, WebP, BMP).
3.  Preprocesses the image data.
4.  Passes the processed image tensor to the trained model's `predict` function.
5.  Returns the classification result (prediction) as a JSON response.

The backend and frontend development for this application were handled by Ahmed Wasim.

---

## 8. Team Members

The RecycleNet project was a collaborative effort by the following team members:

| Team Member | Tasks Completed |
| :--- | :--- |
| **Amr Reffat** | Data preprocessing, ViT model development (with Zaid Ahmed), Data collection. |
| **Zaid Ahmed** | Data preprocessing (with Amr Reffat), Implemented and trained Vision Transformer models, Data collection assistance. |
| **Seif Eldeen Nasser** | Fine-tuned a ResNet model for the classification task. |
| **Omar Shohieb** | Built a handmade/custom model, Model development collaboration (with Mohamed Osama). |
| **Mohamed Osama** | Collaborated with Omar Shohieb to build and refine the handmade/custom model. |
| **Ahmed Wasim** | Developed the backend for the project, Developed the frontend for the project. |
