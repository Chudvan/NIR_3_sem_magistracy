import sqlite3
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import re
import os


def save_to_db(db_path, name_db, df):
    connection = sqlite3.connect(db_path)
    df_columns = [field.replace('-', '_') for field in df.columns]
    df_columns = [field.replace(' ', '_') for field in df_columns]
    try:
        i = df_columns.index('3d_Landmarks')
        df_columns[i] = 'three_d_Landmarks'
    except ValueError:
        pass
    fields = ',\n'.join([f'\t{field} TEXT' for field in df_columns])
    create_costs_table_query = f"""
create table {name_db} (
{fields}
)
"""
    connection.execute(create_costs_table_query)
    connection.commit()
    values = ', '.join(['?' for _ in range(len(df.columns))])
    for row in df.iterrows():
        connection.execute(f"INSERT OR IGNORE INTO {name_db} VALUES({values})", tuple(row[1]))
    connection.commit()
    return connection

def groupby(df, by=None, prediction=2, other=False):
    pa_fields =     [
    'Valence',
    'Arousal'
    ]
    seven_fields = [
        'Neutral', 
        'Happy', 
        'Sad', 
        'Angry', 
        'Surprised', 
        'Scared', 
        'Disgusted'
    ]
    
    if by is None:
        by = pa_fields
        
    df_copy = df[seven_fields + pa_fields].copy()
    
    for field in pa_fields:
        df_copy[field] = df_copy[field].apply(lambda x: round(float(x), prediction))
    for field in seven_fields:
        df_copy[field] = df_copy[field].apply(lambda x: float(x))
    
    df_copy.index = df['Index_']
    
    groupby_fields_sorted = list(sorted(df_copy.groupby(by), key=lambda x: -len(x[1])))
    for group in groupby_fields_sorted:
        for field in seven_fields:
            group[1][field] = round(group[1][field].mean(), prediction)
            
    df_train = pd.DataFrame()
    if other:
        df_other = pd.DataFrame()
    
    for group in groupby_fields_sorted:
        len_group = len(group[1])
        ln_ = np.log10(len_group)
        rand_set = set()
        for _ in range(int(round(ln_, 0)) + 1):
            i = random.randint(0, len_group - 1)
            while i in rand_set:
                i = random.randint(0, len_group - 1)
            rand_set.add(i)
            df_train = pd.concat([df_train, group[1].iloc[i:i + 1]], axis=0)
        if other:
            all_i_without_rand_set = set(range(len_group)) - rand_set
            df_other = pd.concat([df_other, group[1].iloc[list(all_i_without_rand_set)]], axis=0)
    if other:
        return df_train, df_other
    return df_train

def apply_float(df_, columns):
    for field in columns:
        df_[field] = df_[field].apply(lambda el: float(el))
        
def make_valid_df(df_, columns=None):
    if columns is not None:
        apply_float(df_, columns)
    df_.index = df_['Index_']
    
def refitting(models, test, df_metrics, df_train=None, v=1, 
              layer='first', epochs=20, batch_size=20, type_='diff'):
    for nn_tuple in models:
        nn = nn_tuple[2]
        print('refit', nn_tuple[0])
        if type_ == 'diff':
            df_train = nn.create_train_df_from_diff(test)
        elif type_ == 'split' and df_train is not None:
            pass
        else:
            raise Exception('Unknown refitting type.')
        nn.fit(df_train, epochs=epochs, batch_size=batch_size)
        entry_dict = {'model': nn_tuple[1] + f'_{v}', 'layer': layer, 'N': nn_tuple[1]}
        entry_dict.update({metric: nn.model_metric(test, metric) for metric in metrics})
        df_metrics = df_metrics.append(entry_dict, ignore_index = True)
        print(entry_dict)
    return df_metrics

def plot_emotions(models, df_clear, df_metrics, df_clear_metrics, scale=False, figsize=(20, 15)):
    plt.figure(figsize=figsize)
    for i, model_tuple in enumerate(models):
        values = model_tuple[2].predict(df_clear).max().values
        if scale:
            values /= df_clear.max().values[:-2]
        plt.plot(seven_fields, values, label=model_tuple[0])
        entry_dict = {'model': model_tuple[0]}
        entry_dict.update({metric: df_metrics.iloc[i][metric] for metric in metrics})
        entry_dict.update({emotion: values[j] for j, emotion in enumerate(seven_fields)})
        df_clear_metrics = df_clear_metrics.append(entry_dict, ignore_index = True)
    plt.xlabel("Эмоции")
    plt.ylabel("Максимальные значения")
    plt.legend()
    plt.show()
    return df_clear_metrics

def create_add_to_index(csv_file):
    res = []
    without_participant = re.split('Participant \d*', csv_file)[1]
    fragments = re.split('Analysis ', without_participant)
    res.append(fragments[0])
    res.append(re.split('_video_', fragments[1])[0])
    return ''.join(res)

def replace_end_symb(csv_path, encoding=None):
    if not encoding:
        encoding = "UTF-8"
    f = open(csv_path, encoding = encoding)
    text = f.read()
    text = text.replace('\t\n', '\n')
    f.close()
    f = open(csv_path, 'w', encoding = encoding)
    f.write(text)
    f.close()
    
def create_correct_df(dir_, file, encoding=None):
    if not encoding:
        encoding = "UTF-8"
    csv_path = os.path.join(dir_, file)
    replace_end_symb(csv_path, encoding)
    df = pd.read_csv(csv_path, sep='\t', skiprows = lambda i: i in list(range(8)), encoding = encoding)
    if len(df.columns) < 70:
        df = pd.read_csv(csv_path, sep=';', skiprows = lambda i: i in list(range(8)), encoding = encoding)
    df_first = df.columns[0]
    unknown_index = list(df.columns).index('Event Marker')
    df.columns = df.columns[1:].insert(unknown_index, 'UNKNOWN')
    df.insert(0, df_first, df.index)
    add_to_index = create_add_to_index(file)
    df.index = [index + add_to_index for index in df.index]
    return df
