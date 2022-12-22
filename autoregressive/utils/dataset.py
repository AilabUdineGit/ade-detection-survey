import pandas as pd
from typing import Tuple

def load_dataset(corpus: str, split_folder: str, gs: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
        Returns training and test set from the correct corpus and split folder
        if gs (grid search) is true, the test set is the validation set

        Parameters
        ----------
        corpus : str
            where the data come from (smm4h, cadec)
        split_folder : str
            split folder for the ids
        gs : bool
            specify if you are doing the grid search

        Returns
        -------
        (train_ds, test_ds) : Tuple(pd.DataFrame, pd.DataFrame)
            the couple of train and test/validation set
    """
    df = pd.read_pickle(f"assets/datasets/{corpus}.pkl")
    convert = lambda lines : [_id.replace("\n","") for _id in lines if _id != '\n']

    # read ids from the corresponding split folder
    with open(f"assets/splits/{split_folder}/train.id", "r") as fp:
        train_ids = convert(fp.readlines())
    with open(f"assets/splits/{split_folder}/test.id", "r") as fp:
        test_ids = convert(fp.readlines())
    with open(f"assets/splits/{split_folder}/validation.id", "r") as fp:
        validation_ids = convert(fp.readlines())

    if gs:
        test_ds = validation_ids 
    else: 
        train_ids.extend(validation_ids)
    train_ds = df[df.index.isin(train_ids)]
    test_ds = df[df.index.isin(test_ids)]

    if len(train_ds) == 0 or len(test_ds) == 0:
        raise Exception("Wrong IDS for train of test dataset. Check them!")
    
    return train_ds, test_ds
