from utils.data_handler import get_timepoints_data
if __name__ == "__main__":
    import os
    cur_dir = os.path.dirname(__file__)
    project_root = os.path.join(cur_dir, '..')
    data_path = os.path.join(project_root, 'data', 'iTYPE_1_iCE_0007_iCL_2_mi_af10_168_01_SP009a_wf_ALL02_REWTIM_01')
    df = get_timepoints_data(data_path)
    for each in df:
        data = each[1]
        train = []
        test = []
        for record in data.iterrows():
            dat = record[1]
            sample = [int(dat['label']-1), dat['f1'], dat['f2']]
            train.append(sample)
            print(sample)
        # print(each)

