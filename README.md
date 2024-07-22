# 🩺 PIE-Med: Predicting, Interpreting and Explaining Medical Recommendations 

Welcome to the repository for **PIE-Med**, a cutting-edge system designed to enhance medical decision-making through the integration of Graph Neural Networks (GNNs), Explainable AI (XAI) techniques, and Large Language Models (LLMs).

## 🎥 Demo (or GIF)
[Watch our demo](link_to_demo) to see PIE-Med in action and learn how it can transform healthcare recommendations!

## 📊 Data Source
We use the **[MIMIC-III](https://mimic.physionet.org/)** dataset, a freely accessible critical care database containing de-identified health information, including vital signs, laboratory test results, medications, and more. You can find more details about the dataset here:

## 🛠 Technologies Used
- **Python**: Core programming language
- **Pandas**: Data manipulation and analysis ([Pandas Documentation](https://pandas.pydata.org/))
- **PyHealth**: Medical data preprocessing ([PyHealth Documentation](https://pyhealth.readthedocs.io/en/latest/))
- **PyTorch Geometric**: Building and training GNNs ([PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/en/latest/))
- **Integrated Gradients & GNNExplainer**: Interpretability techniques
- **Streamlit**: User interface development ([Streamlit Documentation](https://streamlit.io/))
- **Py AutoGen Multi-Agent Conversation Framework**: Multi-agent collaboration and explanation ([Py AutoGen Documentation](https://microsoft.github.io/autogen/))

## 🔬 Methodological Workflow
PIE-Med follows a comprehensive Predict→Interpret→Explain (PIE) paradigm:

1. **Prediction Phase**: We construct a heterogeneous patient graph from MIMIC-III data and apply GNNs to generate personalized medical recommendations.
2. **Interpretation Phase**: Integrated Gradients and GNNExplainer techniques are used to provide insights into the GNN's decision-making process.
3. **Explanation Phase**: A collaborative ensemble of LLM agents analyzes the model's outputs and generates comprehensive, understandable explanations.

(INSERIRE IMMAGINE)

## 🌟 Key Features
- **Integration of GNNs and LLMs**: Combining structured machine learning with natural language processing for robust recommendations.
- **Enhanced Interpretability**: Using XAI techniques to make the decision-making process transparent.
- **Collaborative Explanation**: Multi-agent LLMs provide detailed and understandable recommendations.

## 🚀 Getting Started
Follow these steps to set up and run PIE-Med on your local machine:

### Prerequisites
Ensure you have the following installed:
- Python 3.7+

### Installation
1. **Clone the repository**:
    ```bash
    git clone https://github.com/picuslab/PIE-Med.git
    cd PIE-Med
    ```

2. **Create a virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application
1. **Run the Streamlit application**:
    ```bash
    streamlit run dashboard.py
    ```

   Open your web browser and go to `http://localhost:8501` to interact with the application.


## 📈 Conclusions
PIE-Med showcases the potential of combining GNNs, XAI, and LLMs to improve medical recommendations, enhancing both accuracy and interpretability. Our system effectively separates prediction from explanation, reducing biases and enhancing decision quality.

## 🙏 Acknowledgments
We extend our gratitude to the creators of the MIMIC-III database, the developers of the Python libraries used, and our research team for their contributions to this project.

👨‍💻 This project was developed by Antonio Romano, Giuseppe Riccio, Marco Postiglione and Vincenzo Moscato
