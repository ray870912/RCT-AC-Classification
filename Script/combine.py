import pandas as pd

aType = ["flexion", "extension", "abduction", "adduction", "horzAbduction", "horzAdduction", "inRotation", "exRotation"]

delete = ['R01','R02','R03','R04','R05','R08','R09','R13','R22','R24',\
          'R30','R43','R49','R50','R54','R61','R67','R70','R72','R73',\
          'A01','A02','A03','A04','A07','A11','A23','A24','A25','A26','A31','A32','A36','A37',\
          'H01','H09','H15','H21','H25','H27','H28','H29','H30','H31','H32','H33']

dataAll = pd.DataFrame()

for j in aType:
    dataTemp = pd.read_csv("/Users/ray/Thesis/Output/" + j + ".csv")
    dataTemp = dataTemp.add_prefix(j + "_")
    # Cleaning
    dataTemp.columns.values[0] = "ID"
    dataTemp = dataTemp[~dataTemp['ID'].str.contains('|'.join(delete))]
    # Merge different groups
    if j == 'flexion':
        dataAll = dataTemp
    else:
        dataAll = pd.merge(dataAll, dataTemp, how='outer', on="ID")

# Merge to Clinical Data
clinic = pd.read_csv('/Users/ray/Thesis/shoulder_all_clinical.csv')
clinic_group = clinic[['ID', 'group']]
# Combine group
dataAll = pd.merge(dataAll, clinic_group, on="ID")

# Imputing
AC_data = dataAll.loc[dataAll['group'] == 'AC'].reset_index()
RCT_data = dataAll.loc[dataAll['group'] == 'RCT'].reset_index()
HC_data = dataAll.loc[dataAll['group'] == 'HC'].reset_index()

AC_data = AC_data.apply(lambda x: x.fillna(x.mean()) if x.dtype.kind in 'biufc' else x.fillna('.'))
RCT_data = RCT_data.apply(lambda y: y.fillna(y.mean()) if y.dtype.kind in 'biufc' else y.fillna('.'))
HC_data = HC_data.apply(lambda z: z.fillna(z.mean()) if z.dtype.kind in 'biufc' else z.fillna('.'))

trainData = pd.concat([AC_data, RCT_data, HC_data], ignore_index=True)
trainData = trainData.reset_index().sort_values(['ID'])
trainData = trainData.iloc[: , 2:]

trainData.to_csv('ShoulderAll_0722.csv', index=False)
        

