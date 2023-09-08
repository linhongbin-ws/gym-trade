import os
def get_csv_dir(root_dir:str):
    """get csv file names"""
    csv_list = []
    for root, dirs, files in os.walk(root_dir, topdown=False):
        for file in files:
            if file.endswith(".csv"):
                file_name = os.path.join(root, file)
                csv_list.append(file_name)
    return csv_list
