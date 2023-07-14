# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 19:47:07 2023

@author: Celian
"""
# libraries
import pandas as pd
import sys
import os
import seaborn as sns
import matplotlib.pyplot as plt

sys.path.append('../')
import configparser

PACKAGE_PARENT = '.'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
PREFIX_PATH = "/".join(os.path.dirname(os.path.abspath(__file__)).split("/")[:-2]) + "/"

from utils import read_data, write_data


def viz_clip_similarity(metrics, df):
    fig, axs = plt.subplots(3, 2, figsize=(20, 20))
    fig.suptitle('Clip Similarity bewteen generated images')
    n=0
    for i in range(3):
        for j in range(2):
            print(n)
            print("----------------")
            print(metrics[n])
            print(df["withoutIMG_"+metrics[n]].describe())
            
            print(metrics[n])
            print(df["withIMG_"+metrics[n]].describe())
            print("----------------")
            #sns.set(style="darkgrid")
            sns.set_style("whitegrid")
            sns.kdeplot(data=df, x="withIMG_"+metrics[n], color="skyblue", label="withIMG",  shade=True,ax=axs[i, j])
            sns.kdeplot(data=df, x="withoutIMG_"+metrics[n], color="red", label="without IMG",shade=True, ax=axs[i, j])
            t=" - ".join(metrics[n].split("-"))
            axs[i, j].set_xlabel(t)
            axs[i, j].legend() 
            n+=1
                
    plt.show()

########### GROUND TRUTH COMPARE
def viz_clip_similarity_ground_truth(parsed_json):

    metrics2=["groundtruth-basic_prompt","groundtruth-dbpedia_abstract_prompt","groundtruth-plain_prompt","groundtruth-verbalised_prompt"]
    data = {}
    for type_ in parsed_json.keys():
        for QID in parsed_json[type_].keys():
            #data["QID"].append(QID)
            for m in metrics2:
                new_col=type_+"_"+m
                if(new_col not in data.keys()):
                        data[new_col]=[]
                if m in parsed_json[type_][QID].keys():
                    data[new_col].append(parsed_json[type_][QID][m])
                else:
                    data[new_col].append(None)
    df = pd.DataFrame(data)
    # fig, axs = plt.subplots(1, 4, figsize=(7, 7))
    sns.set(style="whitegrid")
    colors=["blue","red","green","yellow"]
    for i in range(len(metrics2)):
            sns.kdeplot(data=df, x="withIMG_"+metrics2[i], color=colors[i], label=metrics2[i], shade=False)
        
    plt.legend(loc=(1.04, 0))
    plt.xlabel("CLIP cos sim")
    plt.title("Similarities between generated images and ground truth")
    
    plt.show()
    

def main(parsed_json):
    metrics=["basic_prompt-dbpedia_abstract_prompt","basic_prompt-plain_prompt","basic_prompt-verbalised_prompt","plain_prompt-dbpedia_abstract_prompt","plain_prompt-verbalised_prompt","verbalised_prompt-dbpedia_abstract_prompt"]
    # CHECK LENGTH OF TWO VECT 
    data = {}
    for type_ in parsed_json.keys():
        for QID in parsed_json[type_].keys():
            #data["QID"].append(QID)
            for m in metrics:
                new_col=type_+"_"+m
                if(new_col not in data.keys()):
                        data[new_col]=[]
                if m in parsed_json[type_][QID].keys():
                    data[new_col].append(parsed_json[type_][QID][m])
                else:
                    data[new_col].append(None)
    data_df = pd.DataFrame(data)

    viz_clip_similarity(metrics, data_df)
    viz_clip_similarity_ground_truth(parsed_json)
if __name__=="__main__":
    config = configparser.ConfigParser()
    config.read(PREFIX_PATH + "config.ini")
    CLIP_SD21_dist_img_img_results_path  = PREFIX_PATH + config["EVALUATION"]["CLIP_SD21_dist_img_img_results_path"]
    parsed_json = read_data.read_json(CLIP_SD21_dist_img_img_results_path)
    main(parsed_json)
    
