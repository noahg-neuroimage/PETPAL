import pandas as pd
import numpy as np
import ants
from scipy.stats import ttest_ind
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import tempfile
from petpal.io.table import TableSaver
from petpal.utils.stats import RegionalStats
from petpal.meta.auto_cli import auto_cli
import seaborn as sns


def bids_data_type(input_data_path: str):
    return Path(input_data_path).suffix

class TableLoader:
    @staticmethod
    def load_csv(input_data_path: str) -> pd.DataFrame:
        return pd.read_csv(input_data_path)

    @staticmethod
    def load_tsv(input_data_path: str) -> pd.DataFrame:
        return pd.read_csv(input_data_path, sep=r'\s+')

    @staticmethod
    def load_json(input_data_path: str) -> pd.DataFrame:
        json_series = pd.read_json(input_data_path, typ='series')
        data_path_suffix = input_data_path.split('_')[-1]
        data_type = data_path_suffix.split('.')[0]
        return pd.DataFrame({data_type: json_series}).transpose()

    data_loading_factory = {
        '.csv': load_csv,
        '.tsv': load_tsv,
        '.json': load_json
    }

    def load_data(self, input_data_path: str) -> pd.DataFrame:
        data_path_ext = bids_data_type(input_data_path=input_data_path)
        data_loader = self.data_loading_factory[data_path_ext]
        return data_loader(input_data_path)


class Population:

    table_key = 'table'


    def transpose_table(self, table: pd.DataFrame):
        saver = TableSaver()
        fd, temp = tempfile.mkstemp(suffix='.csv',prefix='PetpalTable')
        saver.save(table.T, temp)
        return pd.read_csv(temp, engine='python', index_col=0)

    def load_pats(self, pats_file: str):
        return pd.read_csv(pats_file, sep=None, engine='python')

    def load_individual_data(self, processed_data_path: Path, divisions: list[str], pat: pd.Series):
        pat_data = {}
        pat_data['participant_id'] = pat['participant_id']
        pat_data['VYr'] = pat['VYr']

        table_loader = TableLoader()
        table = table_loader.load_data(str(processed_data_path))
        #table = pd.read_csv(processed_data_path,sep=r'\s+',engine='python')
        pat_data[self.table_key] = self.transpose_table(table)
        for div in divisions:
            pat_data[div] = pat[div]
        return pat_data

    def population_result(self,
                          pats_file: str,
                          proc_dir: str,
                          file_stem: str,
                          file_category: str,
                          divisions: list[str]):
        pats = self.load_pats(pats_file)
        data = {}
        for pat in pats.iloc:
            prefix = f"sub-{pat['participant_id']}_ses-{pat['VYr']}"
            subdir = Path(proc_dir,f"sub-{pat['participant_id']}",f"ses-{pat['VYr']}",file_category)
            processed_data_path = Path(subdir,f"{prefix}_{file_stem}")
            if processed_data_path.exists():
                data[prefix] = self.load_individual_data(processed_data_path, divisions, pat)
            else:
                print(f"sub-{pat['participant_id']} ses-{pat['VYr']} did not run or failed qc")
        self.data = pd.DataFrame(data).transpose()
        return self.data
    
    def filter_data(self, data, column, val):
        filtered_data = data[data[column]==val]
        return filtered_data

    def retrieve_region_vals(self,
                             filtered_data: pd.DataFrame,
                             region: str,
                             key: str='BP') -> pd.Series:
        tables = filtered_data[self.table_key]
        out_series = pd.Series()
        for i,table in enumerate(tables):
            out_series[tables.index[i]] = table[key][region]
        return out_series

    def filter_twice_and_ttest(self,
                             region: str,
                             filters_first: dict,
                             filters_second: dict,
                             key: str='BP',
                             **kwargs) -> go.Figure:
        data1 = self.data.copy()
        for filter in filters_first:
            data1 = self.filter_data(data1, filter, filters_first[filter])
        vals1 = self.retrieve_region_vals(data1, region, key)
        print(f"Test group 1: {vals1}")
        self.data_group_1 = vals1
        data2 = self.data.copy()
        for filter in filters_first:
            data2 = self.filter_data(data2, filter, filters_second[filter])
        vals2 = self.retrieve_region_vals(data2, region, key)
        print(f"Test group 2: {vals2}")
        self.data_group_2 = vals2

        ttest_result = ttest_ind(vals1, vals2)

        return ttest_result


    def __call__(self,
                 pats_file: str,
                 out_data_path: str,
                 proc_dir: str,
                 file_stem: str,
                 file_category: str,
                 data_key: str,
                 divisions: list[str],
                 regions: list[str]):
        """Load data from individual processed scans into a spreadsheet of results to analyze.
        
        Args:
            pats_file (str): Path to participants.csv
            out_data_path (str): Path to where data table is saved
            proc_dir (str): Path to directory with processed data for all participants
            file_stem (str): Data file naming convention after subject/session prefix
            divisions (list[str]): Data columns to keep from participants.csv
            regions (list[str]): Regions to analyze
        
        Returns:
            population_result (pd.DataFrame): Dataframe with all processed data
        """
        pop_result = self.population_result(pats_file=pats_file,
                                            proc_dir=proc_dir,
                                            file_stem=file_stem,
                                            file_category=file_category,
                                            divisions=divisions)
        data_byregion = pop_result.copy()
        data_byregion = self.filter_data(data_byregion, 'qc', 0)

        data_looper = data_byregion.copy()

        for region in regions:
            
            binding_list = []
            for i, pat in enumerate(data_looper['participant_id']):
                table = data_byregion['table'].iloc[i]
                binding = table[data_key][region]
                binding_list.append(binding)
            data_byregion[region] = binding_list

        data_byregion = data_byregion.drop('table', axis=1)
        data_byregion.to_csv(out_data_path)

def main():
    auto_cli(Population)

if __name__=='__main__':
    main()
