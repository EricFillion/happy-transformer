import pandas as pd
from tqdm import tqdm_notebook

prefix = 'data/'


train_df = pd.read_csv(prefix + 'train.csv', header=None)
train_df = train_df.head(100)


test_df = pd.read_csv(prefix + 'test.csv', header=None)
test_df = test_df.head(100)


train_df[0] = (train_df[0] == 2).astype(int)
test_df[0] = (test_df[0] == 2).astype(int)


train_df = pd.DataFrame({
    'id':range(len(train_df)),
    'label':train_df[0],
    'alpha':['a']*train_df.shape[0],
    'text': train_df[1].replace(r'\n', ' ', regex=True)
})

train_df.head()



dev_df = pd.DataFrame({
    'id':range(len(test_df)),
    'label':test_df[0],
    'alpha':['a']*test_df.shape[0],
    'text': test_df[1].replace(r'\n', ' ', regex=True)
})

dev_df.head()


train_df.to_csv('data/train.tsv', sep='\t', index=False, header=False, columns=train_df.columns)
dev_df.to_csv('data/dev.tsv', sep='\t', index=False, header=False, columns=dev_df.columns)