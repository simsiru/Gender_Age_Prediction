import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import torch


def plot_metrics(path: str) -> None:
    fig, ax = plt.subplots(1, 5, figsize=(14,3))

    df = pd.read_csv(path)
    df = df.set_index('epoch')
    
    for i, metric in enumerate(['loss','auroc','acc','rmse','r2']):
        sns.lineplot(data=df[[f'train/{metric}', f'val/{metric}']], ax=ax[i])
        ax[i].set_ylabel(metric)

    fig.tight_layout()
    plt.show()


def process_output(output_tensor: torch.tensor, threshold: float=0.5) -> dict:
    final_pred_dict = defaultdict(list)

    for pred_batch in output_tensor:
        for label in pred_batch[0]:
            final_pred_dict['age_pred'].append(int(label[0]))

        gender_batch_list = (
            torch.nn.functional.sigmoid(pred_batch[1]) > threshold
        ).int().tolist()

        for label in gender_batch_list:
            final_pred_dict['gender_pred'].append(label[0])
            
        for label in torch.nn.functional.sigmoid(pred_batch[1]).tolist():
            final_pred_dict['gender_pred_prob'].append(label[0])
        
    return final_pred_dict


def display_samples(samples: pd.DataFrame) -> None:
    race_dict = {
        0:'White',
        1:'Black',
        2:'Asian',
        3:'Indian',
        4:'Others like Hispanic, Latino, Middle Eastern'
    }
        
    gender_dict = {
        0:'Male',
        1:'Female'
    }
    for sample in samples.iterrows():
        img = plt.imread(sample[1]['path'])

        plt.imshow(img)
        plt.show()

        print(f"True age: {sample[1]['age']}, Predicted age: {sample[1]['age_pred']}")
        print(f"True gender: {gender_dict[int(sample[1]['gender'])]}, ",
              f"Predicted gender: {gender_dict[int(sample[1]['gender_pred'])]}")
        print(f"Race: {race_dict[int(sample[1]['race'])]}\n")