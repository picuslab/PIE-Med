import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import torch
from typing import Tuple, List

from pyhealth.medcode import InnerMap
from pyhealth.datasets import MIMIC3Dataset, SampleEHRDataset
from pyhealth.tasks import medication_recommendation_mimic3_fn, diagnosis_prediction_mimic3_fn
from pyhealth.models import GNN
from pyhealth.explainer import HeteroGraphExplainer

@st.cache_resource(hash_funcs={torch.nn.parameter.Parameter: lambda _: None})
def load_gnn() -> Tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module, torch.nn.Module,
                        MIMIC3Dataset, SampleEHRDataset, SampleEHRDataset]:
    dataset = MIMIC3Dataset(
        root=st.secrets.s3.s3_uri,
        tables=["DIAGNOSES_ICD","PROCEDURES_ICD","PRESCRIPTIONS","NOTEEVENTS_ICD"],
        code_mapping={"NDC": ("ATC", {"target_kwargs": {"level": 4}})},
    )

    mimic3sample_med = dataset.set_task(task_fn=medication_recommendation_mimic3_fn)
    mimic3sample_diag = dataset.set_task(task_fn=diagnosis_prediction_mimic3_fn)

    model_med_ig = GNN(
        dataset=mimic3sample_med,
        convlayer="GraphConv",
        feature_keys=["procedures", "diagnosis", "symptoms"],
        label_key="medications",
        k=0,
        embedding_dim=128,
        hidden_channels=128
    )

    model_med_gnn = GNN(
        dataset=mimic3sample_med,
        convlayer="GraphConv",
        feature_keys=["procedures", "diagnosis", "symptoms"],
        label_key="medications",
        k=0,
        embedding_dim=128,
        hidden_channels=128
    )

    model_diag_ig = GNN(
        dataset=mimic3sample_diag,
        convlayer="GraphConv",
        feature_keys=["procedures", "medications", "symptoms"],
        label_key="diagnosis",
        k=0,
        embedding_dim=128,
        hidden_channels=128
    )

    model_diag_gnn = GNN(
        dataset=mimic3sample_diag,
        convlayer="GraphConv",
        feature_keys=["procedures", "medications", "symptoms"],
        label_key="diagnosis",
        k=0,
        embedding_dim=128,
        hidden_channels=128
    )

    return model_med_ig, model_med_gnn, model_diag_ig, model_diag_gnn, dataset, mimic3sample_med, mimic3sample_diag

@st.cache_data(hash_funcs={torch.Tensor: lambda _: None})
def get_list_output(y_prob: torch.Tensor, last_visit: pd.DataFrame, task: str, _mimic3sample: SampleEHRDataset, 
                   top_k: int = 10) -> List[str]:
    sorted_indices = []
    for i in range(len(y_prob)):
        top_indices = np.argsort(-y_prob[i, :])[:top_k]
        sorted_indices.append(top_indices)

    list_output = []

    # get the list of all labels in the dataset
    if task == "medications":
        list_labels = _mimic3sample.get_all_tokens('medications')
        atc = InnerMap.load("ATC")
    elif task == "diagnosis":
        list_labels = _mimic3sample.get_all_tokens('diagnosis')
        icd9 = InnerMap.load("ICD9CM")

    sorted_indices = list(sorted_indices)
    # iterate over the top indexes for each sample in test_ds
    for (i, sample), top in zip(last_visit.iterrows(), sorted_indices):
        # create an empty list to store the recommended medications for this sample
        sample_list_output = []

        # iterate over the top indexes for this sample
        for k in top:
            # append the medication at the i-th index to the recommended medications list for this sample
            if task == "medications":
                sample_list_output.append(atc.lookup(list_labels[k]))
            elif task == "diagnosis":
                if list_labels[k].startswith("E"):
                    list_labels[k] = list_labels[k] + "0"
                sample_list_output.append(icd9.lookup(list_labels[k]))

        # append the recommended medications for this sample to the recommended medications list
        list_output.append(sample_list_output)

    return list_output, sorted_indices

def explainability(model: GNN, explain_dataset: SampleEHRDataset, selected_idx: int, 
                   visualization: str, algorithm: str, task: str, threshold: int):
    explainer = HeteroGraphExplainer(
        algorithm=algorithm,
        dataset=explain_dataset,
        model=model,
        label_key=task,
        threshold_value=threshold,
        top_k=threshold,
        feat_size=128,
        root="./streamlit_results/",
    )

    if task == "medications":
        visit_drug = explainer.subgraph['visit', 'medication'].edge_index
        visit_drug = visit_drug.T

        n = 0
        for vis_drug in visit_drug:
            vis_drug = np.array(vis_drug)
            if vis_drug[1] == selected_idx:
                break
            n += 1
    elif task == "diagnosis":
        visit_diag = explainer.subgraph['visit', 'diagnosis'].edge_index
        visit_diag = visit_diag.T

        n = 0
        for vis_diag in visit_diag:
            vis_diag = np.array(vis_diag)
            if vis_diag[1] == selected_idx:
                break
            n += 1

    explainer.explain(n=n)
    if visualization == "Explainable":
        explainer.explain_graph(k=0, human_readable=True, dashboard=True)
    else:
        explainer.explain_graph(k=0, human_readable=False, dashboard=True)

    explainer.explain_results(n=n)
    explainer.explain_results(n=n, doctor_type="Internist_Doctor")

    HtmlFile = open("streamlit_results/explain_graph.html", 'r', encoding='utf-8')
    source_code = HtmlFile.read() 
    components.html(source_code, height=520)