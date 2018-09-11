# -*- coding: utf-8 -*-
"""
Patient Model built from https://physionet.org/challenge/2012/#weight

Outcome Info:
Survival > Length of stay  ⇒  Survivor
Survival = -1  ⇒  Survivor
2 ≤ Survival ≤ Length of stay  ⇒  In-hospital death

Data Info:
RecordID (a unique integer for each ICU stay)
Age (years)
Height (cm)
Weight (kg)
Gender  0: Female
        1: Male
ICUType 1: Coronary Care Unit
        2: Cardiac Surgery Recovery Unit
        3: Medical ICU
        4: Surgical ICU


@author: Theo Fountain III
"""
from pathlib import Path
import pickle
import pdb

import pandas as pd


# import matplotlib.pyplot as plt
# import matplotlib.ticker as mticker
# import matplotlib.dates as mdates
# import matplotlib


# Bad Data
# 140501 missing vital signs
# 140936 missing vital signs
# 141264 missing vital signs

class MissingData(Exception):
    pass


def patient_info(df):
    """ Return record header  """

    icu_type = {
        1: 'Coronary Care Unit',
        2: 'Cardiac Surgery Recovery Unit',
        3: 'Medical ICU',
        4: 'Surgical ICU'}

    col_labels = ['RecordID', 'Age', 'Gender', 'Height', 'ICUType', 'Weight']
    df_pat = df.drop('Time', axis=1)
    df_pat = df_pat.loc[df_pat.Parameter.isin(values=col_labels)]
    df_pat.drop_duplicates(subset='Parameter', inplace=True)
    df_pat.index = [0] * len(df_pat)
    df_pat.Value = df_pat.Value.apply(int)
    df_pat = df_pat.pivot(index=None, columns='Parameter', values='Value')
    df_pat.set_index('RecordID', inplace=True)
    df_pat.rename(colums={'Height': 'Height(cm)', 'Weight': '(kg)'})
    df_pat.replace(to_replace={-1: None}, inplace=True)
    df_pat.replace(to_replace={
        'Gender': {0: 'Female', 1: 'Male'},
        'ICUType': icu_type}, inplace=True)

    return df_pat


def vital_signs(df, multi_index=None):
    """ """

    rec_id = int(df[df.Parameter == 'RecordID'].Value.tolist()[0])
    df_vs = df[~(df.Parameter.isin([
        'RecordID', 'Age', 'Gender', 'Height', 'ICUType']))]
    df_vs.reset_index(drop=True, inplace=True)
    df_vs = pd.pivot_table(
        df_vs,
        index='Time', columns='Parameter', values='Value')
    df_vs.replace(to_replace={-1: None}, inplace=True)
    df_vs.index = df_vs.index + ':00'
    df_vs.index = pd.to_timedelta(df_vs.index)
    # pdb.set_trace()
    df_vs.rename(columns={'Weight': 'Weight(kg)'})

# Find measuresments taken at same time
    data = df_vs.isnull().describe()
    groups = list(set(zip(*data.loc['top':'freq'].values.tolist())))
    data = data.T
    col_grouped = [
        df_vs.columns[(data.top == each[0]) & (data.freq == each[1])].tolist()
        for each in groups]
    col_list = [e for each in col_grouped for e in each ]
# TODO: Replace with more specific labels
    levels = [
        f'Label_{ctr}' for ctr, each in enumerate(col_grouped)]
    col_index = [
        list(zip([each[0]] * len(each[1]), each[1]))
        for each in list(zip(levels, col_grouped))]
    test = [e for each in col_index for e in each]
    df_vs = df_vs[col_list]
    col_m_index = pd.MultiIndex.from_tuples(test)
    df_vs.columns = col_m_index

    pdb.set_trace()
    # data.loc['top':'freq'].isin()
    # df_vs[~(df_vs.NIMAP.isnull())]

    if multi_index is not None:
        arrays = [len(df_vs.index.tolist()) * (rec_id, ), df_vs.index.tolist()]
        tuples = list(zip(*arrays))
        try:
            multi_index = pd.MultiIndex.from_tuples(
                tuples, names=['Record', 'Time'])
            df_vs.set_index(multi_index, inplace=True)
        except TypeError:
            raise(MissingData(f'{rec_id} missing vital signs'))
            return
    else:
        df_vs.name = rec_id

        return df_vs


def process_frames(file):
    """ """

    with open(file) as f:
            df = pd.read_csv(f)

    pat_info = patient_info(df)
    vitals = vital_signs(df)
    if vitals is not None:
        return {'patient': pat_info, 'vitals': vitals}


def store_frames():
    """Return pickled dataframes"""

    p = Path(r'C:\Users\Theo\Google Drive\Python Projects\Patient Model')
    txt_files = p.glob('**/*.txt')
    frames = [process_frames(f) for f in txt_files]
    patient_db = pd.concat(
        [f['patient'] for f in frames if f is not None],
        sort=False)
    vitals_db = pd.concat(
        [f['vitals'] for f in frames if f is not None],
        sort=False)

    with open('Vitals_A.pickle', 'wb') as p:
        pickle.dump(vitals_db, p)

    with open('Records_A.pickle', 'wb') as p:
        pickle.dump(patient_db, p)


def load_records():

    with open('Vitals_A.pickle', 'rb') as p:
        vitals_db = pickle.load(p)

    with open('Records_A.pickle', 'rb') as p:
        patient_db = pickle.load(p)

    return vitals_db, patient_db


with open('132539.txt') as f:
    df = pd.read_csv(f)

vitals = vital_signs(df)

# def main():
# # store_records()
# # vitals, patients = load_records()


# if __name__ == 'main':
#     main()
