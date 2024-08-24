import argparse
import os
import pandas as pd
import numpy as np
from datasets import concatenate_datasets
from utils.data import *

def main():
    parser = argparse.ArgumentParser(description="Process data.")

    # Adding arguments
    parser.add_argument('--data_folder', type=str, required=True, help='path to data')
    parser.add_argument('--output_folder', type=str, help='path of output dir',default='./')
    parser.add_argument('--processor_path', type=str, required=True,help='path to pre trained processor')

    # Parsing arguments
    args = parser.parse_args()

    # Accessing arguments
    print(f"data_folder: {args.data_folder}")
    print(f"output_folder: {args.output_folder}")
    print(f"processor_path: {args.processor_path}")
    

    assert os.path.exists(args.data_folder), f"folder chosen : {args.data_folder} not found"
    
    
    # parse folder images
    all_paths_imagess = parse_data_paths(args.data_folder)
    # labels
    all_labels = [os.path.basename(os.path.dirname(x)) for x in all_paths_imagess]
    
    df = pd.DataFrame({'path_order':all_paths_imagess,
                  'label':all_labels}
                 )
    
    
    csv_file_save = os.path.join(args.output_folder,'train.csv')
    df.to_csv(csv_file_save,index=False)
    
    data_files = {"train": csv_file_save}
    
    datasets = load_dataset('csv', data_files=data_files)
    
    
    data_to_process = datasets["train"]
    
    processor = LayoutLMv3Processor.from_pretrained(args.processor_path)
    
    
    # we need to define custom features
    features = FEATURES_PROCESSOR['v3']
    
    # extract data to pandas
    df_to_paralelize = data_to_process.to_pandas()
    
    # transform enlarge image for ocr
    transformation = transforms.Compose([transforms.Resize(3000)
                                ])
    
    device: str = "cpu"
    cores: int = None
    splits: int = None
    col_paral: str = "path_order"
    chunks_data = 5
    
    
    if cores is None:
        cores = cpu_count() - 1
    
    if splits is None:
        splits = cores
    
    # split data to chunks
    chunks = np.array_split(np.arange(len(data_to_process)),indices_or_sections = chunks_data)
    
    # path save tensor data
    data_tensor_output = os.path.join(args.output_folder,'tensor_data')
    
    
    # multiprocess chunke data
    for jj,set_c in enumerate(chunks):
        
    
        print(jj)
        
        data_to_process_SAMPE = data_to_process.select(set_c)
        
        
        df_to_paralelize = data_to_process_SAMPE.to_pandas()
        
        processed_data = paralelize_df(
            df_to_paralelize,
            col_func,
            apply_tesser,
            col_paral,
            part=splits,
            cor=splits,
            processor=processor,
            transformation=transformation
        )
        
        data = {i: [] for i in processed_data[col_paral].values[0].keys()}
    
        for j in processed_data[col_paral].values:
    
            [data[key].append(j[key][0]) for key in j]
    
        # load data from dictionary into datasets.arrow_dataset.Dataset object
        processed_data = data_to_process_SAMPE.from_dict(data, features)
    
        # add label collum
        processed_data = processed_data.add_column(
            name="labels", column=data_to_process_SAMPE["label"]
        )
    
        # put data to device
        processed_data.set_format(type="torch", device=device)
        
        path_save = f'part_{jj}'
    
        folder_path_save = os.path.join(data_tensor_output,path_save)
    
        mkdir(folder_path_save)
    
        processed_data.save_to_disk(folder_path_save)
    
    
    
    # pats of saved tesnor data
    all_data = fdir(data_tensor_output,
                    specific_end=None,
                    sepcific_include='part',
                    sepcific_remove='zip')
    
    
    
    
    # sort data parts load collate and merge to single tensor dataset
    nums = [(l,int(o.split('_')[-1])) for l,o in enumerate(all_data)]
    sorted_list = sorted(nums,key=lambda x: x[1])
    sorted_list_path = [all_data[k[0]] for k in sorted_list]
    
    data_to_collate = []
    
    for j in sorted_list_path:
        data_to_collate.append(arrow_dataset.Dataset.load_from_disk(j))
        
    
    merged_tensors = concatenate_datasets(data_to_collate)
    
    # save tensor data
    name_saved_collated = os.path.join(args.output_folder, 'colated_tensors')
    
    mkdir(name_saved_collated)
    
    
    merged_tensors.save_to_disk(name_saved_collated)


if __name__ == '__main__':
    
    main()
    