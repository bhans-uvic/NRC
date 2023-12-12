import yaml
import pandas as pd
import os
import numpy as np

dir = os.path.dirname(os.path.abspath(__file__))

with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

dataDir = dir + config['data']['dataDir']

def getFgFile(fgPath,locDir,ext,bString):
    fgDir = locDir + fgPath
    fgFile = fgPath[1:] + ext
    for f in os.listdir(fgDir):
        if f.endswith(ext) == False:
            metaDataBreak = -1
            with open(fgDir + '/' + f,'r') as fp:
                Lines = fp.readlines()
                for i in range(len(Lines)):
                    if Lines[i].startswith(bString):
                        metaDataBreak = i
                        break
                if metaDataBreak != -1:
                    if os.path.exists(fgDir + '/' + fgFile):
                        with open(fgDir + '/' + fgFile,'a+') as fgf:
                            if fgf.readline() == Lines[i-1][1:]:
                                fgf.writelines(Lines[i+1:])
                            else:
                                print('Mismatching Headers for files:\n',fgDir + '/' + fgFile,'\n' ,fgDir + '/' + f)
                    else:
                        with open(fgDir + '/' + fgFile,'a') as fgf:
                            fgf.write(Lines[i-1][1:].replace('"', '').replace(', ',','))
                            fgf.writelines(Lines[i+1:])
                else:
                    print('Meta Data clean break point not found for file: ',fgDir + '/' + f)
    return fgDir,fgFile

for loc in config['data']['locations']:
    print('*********************************************', loc['path'][1:],'*********************************************')
    locDir = dataDir + loc['path']
    for fgPath in loc['featureGroupPaths']:
        print('\n*******************', fgPath[1:],'*******************')
        print('\t- Removing metadata...')
        fgDir,fgFile = getFgFile(fgPath,locDir,loc['ext'],loc['breakString'])
        df = pd.read_csv(fgDir + '/' + fgFile)
        print('\t-',fgPath[1:],' Data before Cleaning')
        print(df.info())
        print(df.describe())
        print('\t- Performing Feature Group level claening...')
        df.loc[:, df.columns.difference([loc['oldControlCol']])] = df.loc[:, df.columns.difference([loc['oldControlCol']])].apply(pd.to_numeric, errors='coerce')
        df[loc['oldControlCol']] = pd.to_datetime(df[loc['oldControlCol']])
        df[loc['newControlCol']] = df[loc['oldControlCol']].dt.strftime("%Y-%m-%dT%H")
        df = df.drop([loc['oldControlCol']], axis=1)
        for col in df.columns:
            if any(map(col.__contains__, loc['removeStringCol'])):
                df = df.drop([col], axis=1)
        dfGrp = df.groupby(by=[loc['newControlCol']]).mean().reset_index()
        print('\t-',fgPath[1:],' Data After Cleaning')
        print(dfGrp.info())
        print(dfGrp.describe())
        dfGrp.to_csv(locDir + loc['groupedPath'] + '/' + fgFile.split('.')[0] + '_Grouped.csv',index=False)

for loc in config['data']['locations']:
    locGrpDir = dataDir + loc['path'] + loc['groupedPath']
    dfs = {}
    print('\tMerging files...')
    for f in os.listdir(locGrpDir):
        df_name = f.split('.')[0]
        dfs[df_name] = pd.read_csv(locGrpDir + '/' + f)
    merged_df = None
    for k,v in dfs.items():
        if merged_df is None:
            merged_df = v
        else:
            merged_df = pd.merge(merged_df,v,on=loc['newControlCol'], how=loc['mergeMethod'])
    
    column_pairs = [(col, col[:-2]+'_y') for col in df.columns if col.endswith('_x')]

    def merge_and_mean(row):
        for col_x, col_y in column_pairs:
            if not pd.isnull(row[col_x]) and not pd.isnull(row[col_y]):
                row[col_x[:-2]] = np.mean([row[col_x], row[col_y]])
            elif pd.isnull(row[col_x]) and not pd.isnull(row[col_y]):
                row[col_x[:-2]] = row[col_y]
            elif pd.isnull(row[col_y]) and not pd.isnull(row[col_x]):
                row[col_x[:-2]] = row[col_x]
            else:
                row[col_x[:-2]] = np.nan
        return row

    merged_df = merged_df.apply(merge_and_mean, axis=1)
    merged_df = merged_df.drop([col for pair in column_pairs for col in pair], axis=1)
    null_count = merged_df.isnull().sum(axis=1)
    threshold = 2
    merged_df = merged_df[null_count <= threshold]
    merged_df = merged_df.drop(columns = loc['mergedFactorRemove'])
    merged_df = merged_df.dropna(subset=[loc['targetPrm']])
    print('\t ',loc['path'][1:],' Data After Merging:')
    print(merged_df.info())
    print(merged_df.describe())
    print('Data Acquired Between:',merged_df[loc['newControlCol']].min(),' and ',merged_df[loc['newControlCol']].max())
    merged_df.to_csv(dataDir + loc['path'] + '/' + loc['path'][1:] + '_merged.csv',index=False)  
        