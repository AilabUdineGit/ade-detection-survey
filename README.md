# ADE detection - survey

# Autoencoding models

```
cd autoencoding
```

## Database creation

```
python ade_detection.py -i
```

## Run Grid Search

```
python ade_detection.py -<architecture> -<dataset> -gs --run <model>.json  
```

where `<architecture> = {bert_wrapper, bert_crf, bert_lstm}`, `<dataset> = {cadec, smm4h20}`, and `<model>` can be every model mentioned in the paper (in upper case).

## Run Evaluation

```
python ade_detection.py -<architecture> -<dataset> -ft --run <model>.json  
```


# Autoregressive models

```
cd autoregressive
```

## Run Grid Search

```
python main.py -<dataset> -gs -<model>
```

where `<model> = {gpt2, t5, pegasus, bart, scifive}`.


## Run Evaluation 

```
python main.py -<dataset> -ft -<model>
```

# TODO

 - [  ] Merge autoregressive and autoencoding
 - [  ] Add links to download the two datasets
