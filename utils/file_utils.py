import pandas as pd

def dic_to_csv(dic, path, start_ind=0, index_label='id'):
        data_frame = pd.DataFrame(data=dic, index=range(start_ind, start_ind + len(list(dic.values())[0])))
        data_frame.to_csv(path, index_label=index_label)

class CSV_Writer:
    def __init__(self, path, columns):
        self.path = path
        self.dic = dict()
        self.columns = columns
        for x in columns:
            self.dic[x] = []
    
    def add_record(self, record):
        for i, x in enumerate(self.columns):
            self.dic[x].append(record[i])
        self._save()
    
    def _save(self):
        dic_to_csv(self.dic, self.path)
