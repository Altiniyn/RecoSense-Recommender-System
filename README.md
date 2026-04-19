# 🧠 RecoSense: Intelligent Recommender System

**RecoSense** is a sophisticated, full-stack recommendation engine dashboard developed for the **AIE425** course at **Alamein International University**. Built with **Streamlit**, it provides a high-performance, interactive UI to explore, build, and evaluate multiple recommendation strategies using the Amazon Reviews dataset.

---

## 🚀 Key Features

### 1. Multi-Engine Recommendation Architecture
The system implements three primary recommendation paradigms, offering a total of 9 different algorithms:
*   **🤝 Collaborative Filtering**: 
    *   User-Based KNN (Cosine Similarity)
    *   Item-Based KNN (Cosine + Mean Centering)
    *   SVD (Matrix Factorization)
    *   Slope One (Deviation-based)
*   **📄 Content-Based Filtering**:
    *   TF-IDF + Cosine Similarity
    *   Score-Weighted TF-IDF Profiles
*   **🧩 Knowledge-Based Filtering**:
    *   Score-Ranked
    *   Helpfulness-Ranked
    *   Popularity-Ranked

### 2. Advanced Evaluation Suite
Measure model performance across 6 industry-standard metrics:
*   **Precision @ K** & **Recall @ K**
*   **Catalog Coverage**: Percentage of items the system can recommend.
*   **Novelty**: Ability to suggest "long-tail" items that users haven't seen.
*   **Personalization**: How unique recommendations are for different users.

### 3. Modern Interactive UI
*   **Dark-themed Professional Dashboard**: Custom CSS with smooth animations and fluid layouts.
*   **Real-time Analytics**: Interactive Plotly charts for score distributions and user activity.
*   **Dynamic Model Training**: Adjust sample sizes (from 5K to 100K rows) and rebuild engines on the fly.

---

## 🛠️ Tech Stack

| Category | Tools/Libraries |
| :--- | :--- |
| **Frontend** | Streamlit, Custom CSS3 Animations, HTML5 |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Plotly (Graph Objects & Express) |
| **Machine Learning** | Scikit-learn (TF-IDF, KNN), Surprise (SVD, Slope One) |

---

## 📦 Installation & Setup

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd RecoSense
```

### 2. Install Dependencies
Ensure you have Python 3.8+ installed, then run:
```bash
pip install streamlit pandas numpy plotly scikit-learn scikit-surprise
```

### 3. Data Preparation
1. Create a folder named `data/` in the project root.
2. Place your `Reviews.csv` (Amazon Reviews dataset) inside the `data/` folder.

### 4. Run the Application
```bash
streamlit run app.py
```

---

## 📂 Project Structure

```text
├── app.py                # Main Streamlit application & UI logic
├── data/                 
│   ├── loader.py         # Data ingestion & preprocessing
│   └── Reviews.csv       # Dataset (User-provided)
├── recommenders/         
│   ├── collaborative.py  # KNN, SVD, and Slope One logic
│   ├── content_based.py  # TF-IDF and content profiling
│   └── knowledge_based.py# Heuristic-based ranking
├── evaluation/           
│   └── evaluator.py      # Performance metrics calculation
└── README.md             # Project documentation
```

---

## 🎓 Academic Context
*   **Course**: AIE425 - Recommender Systems
*   **Institution**: Alamein International University (AIU)
*   **Faculty**: Artificial Intelligence Engineering

---

## 📄 License
This project is developed for academic purposes. Please cite the authors if you use this code in your research.

---
*Developed with ❤️ by the RecoSense Team.*
