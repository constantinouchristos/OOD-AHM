import os 
import pandas as pd
import numpy as np
import logging
from typing import List
from abc import ABC
import pandas as pd
import re
import string
from datasets import load_dataset, dataset_dict, arrow_dataset
from transformers import LayoutLMv2Processor, LayoutLMv3Processor
from datasets import Features, Sequence, Value, Array2D, Array3D
import datasets
import argparse
from multiprocessing import cpu_count
from torchvision import transforms
from multiprocessing import Pool, cpu_count
import functools
from PIL import Image
import json



FEATURES_PROCESSOR = {
    "v2": Features(
        {
            "image": Array3D(dtype="int64", shape=(3, 224, 224)),
            "input_ids": Sequence(feature=Value(dtype="int64")),
            "attention_mask": Sequence(Value(dtype="int64")),
            "token_type_ids": Sequence(Value(dtype="int64")),
            "bbox": Array2D(dtype="int64", shape=(512, 4)),
        }
    ),
    "v3": Features(
        {
            "input_ids": Sequence(feature=Value(dtype="int64")),
            "attention_mask": Sequence(Value(dtype="int64")),
            "token_type_ids": Sequence(Value(dtype="int64")),
            "bbox": Array2D(dtype="int64", shape=(512, 4)),
            "pixel_values": Array3D(dtype="float64", shape=(3, 224, 224)),
        }
    ),
}


def parse_data_paths(path_input):
    
    all_paths_dirs = [os.path.join(path_input,i) for i in os.listdir(path_input)]
    all_paths_imagess = []

    for i in all_paths_dirs:
        all_paths_imagess += [os.path.join(i,ii) for ii in os.listdir(i)]
        
    all_paths_imagess = [i for i in all_paths_imagess if i.endswith('.png')]
    
    return all_paths_imagess 



def paralelize_df(data, col_func, sub_func, col, part, cor, **kwargs):

    """
    Performs multi processing on data

    Arguments:
        data    :  (pd.DataFrame)  : datset to process
        col_func:  (function)      : function to operate on column of dataframe
        col     :  (str)           : column name
        part    :  (int)           : number od parts to split data
        cor     :  (int)           : number od cores to use
        kwargs  :  (dict)          : additional keyword arguments required for functions
    Returns:
        data    :  (pd.DataFrame)  : processed
    """

    data_split = np.array_split(data, part)
    pool = Pool(cor)

    data = pd.concat(
        pool.map(
            functools.partial(col_func, sub_func=sub_func, col=col, **kwargs),
            data_split,
        )
    )

    pool.close()
    pool.join()

    return data


def col_func(data, sub_func, col, **kwargs):

    """
    Applies a function on a the column col of the dataframe

    Arguments:
        data    :  (pd.DataFrame)  : datset to process
        sub_func:  (function)      : function to operate on column of dataframe
        col     :  (str)           : column name
        kwargs  :  (dict)          : additional keyword arguments required for functions
    Returns:
        data    :  (pd.DataFrame)  : processed
    """

    data[col] = data[col].apply(functools.partial(sub_func, **kwargs))

    return data



def apply_tesser(x, **kwargs):

    """
    processing function

    Arguments:
        x       :  (any)     : item to apply processing on
        kwargs  :  (dict)    : additional keyword arguments required for functions
    Returns:
        encoded_inputs    :  processed output
    """

    image = Image.open(x).convert("RGB")
    
    

    dl_processor = kwargs.get("processor")
    transformation = kwargs.get("transformation")
    
    image = transformation(image)

    encoded_inputs = dl_processor(
        image, padding="max_length", truncation=True, return_token_type_ids=True
    )

    return encoded_inputs


def write_df_file(df,path) -> None:

    path_temp_to_save = os.path.join(self.args.output_dir, "temp")

    if not os.path.exists(path_temp_to_save):
        os.makedirs(path_temp_to_save)

    self.path_temp_to_file = os.path.join(path_temp_to_save, "train.csv")
    self.df.to_csv(self.path_temp_to_file, index=False)


def fdir(path,specific_end='.zip',sepcific_remove=None,sepcific_include=None):
    
    out = [os.path.join(path,i) for i in os.listdir(path)]

    if specific_end is not None:
        out = [i for i in out if i.endswith(specific_end)]
        
    if sepcific_remove is not None:
        out = [i for i in out if sepcific_remove not in i]
        
    if sepcific_include is not None:
        out = [i for i in out if sepcific_include in i]
    
    return out


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def json_serialize(dicto):

    new_dict = {}

    for i,j in dicto.items():
        # print(type(i),type(j))
        if isinstance(j,dict):
            new_dict[i] = json_serialize(j)

        elif isinstance(j,list) or isinstance(j,np.array):

            new_dict[i] = [int(o) for o in j]

        else:
            if isinsance(j,int) or isinsance(j,int):
                new_dict[i] = int(j)

            else:
                new_dict[i] = j

    return new_dict
    


def write_json(file_name_path,data):
    with open(file_name_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


