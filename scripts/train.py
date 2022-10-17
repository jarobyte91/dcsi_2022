"""
BERT embeddings + Linear
"""
import os
# os.environ['TRANSFORMERS_OFFLINE'] = '1'


import pandas as pd

from functools import partial
from pathlib import Path

from tokenizers import Tokenizer
from transformers import AutoTokenizer, AutoModel
import transformers
from transformers.models.auto.tokenization_auto import logger

import random

import torch
import torchmetrics
import torch.nn as nn
from torch.utils import data
from torch.utils.data import Dataset, DataLoader, random_split

import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, StochasticWeightAveraging
from pytorch_lightning.plugins import DDPPlugin

transformers.logging.set_verbosity_error()
# TRANSFORMER_CACHE = Path("/home/amanjais/projects/def-emilios/dsci_2022/resources/transformer_cache")
TRANSFORMER_CACHE = Path("resources/transformer_cache")



from dcsi_2022.chunksumm_vectorized import SuMM_with_tokenizer,SummDataModule,CHUNKSUMM_CONV #


def train(data_params: dict,trainData,valData,testData,only_test_epoch=False):

    # Printing params
    print("PARAMETERS \n")
    for k, v in data_params.items():
        print(f"{k} : {v} ")

    pretrained_model_name = data_params['BACKBONE']

    # Tokenizer--------------------------------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name,cache_dir=TRANSFORMER_CACHE,)
    print("Tokenizer Loaded...\n")

    # Datamodule --------------------------------------------------------------------------------


    data_module = SummDataModule(
    trainData = trainData, 
    valData = valData,
    testData = testData,
    tokenizer = tokenizer,
    workers = 4,
    batch_size = data_params['BATCH_SIZE'],
    SuMMDataset=SuMM_with_tokenizer,
    train_size=data_params['train-size'],
    process_paper_level=data_params['process_paper_level'])
    data_module.setup()


    print("Dataset loaded in DataModule...\n")



    # Bert-Classifier --------------------------------------------------------------------------------


    pretrained_model = AutoModel.from_pretrained(
        pretrained_model_name,
        cache_dir=TRANSFORMER_CACHE,
    )

    classifier = CHUNKSUMM_CONV(pretrained_model,tokenizer,learning_rate = 1e-5,n_classes=data_params['N_CLASS'],enable_chunk=True,freeze_bert=data_params['freeze_bert'])

    # Callbacks --------------------------------------------------------------------------------

    
    # logger = TensorBoardLogger("lightning_logs", name=experiment_name)
    experiment_name = data_params["EXP_NAME"]

    logger = MLFlowLogger(experiment_name=experiment_name, run_name=data_params['RUN_NAME'],
                          tracking_uri="file:./mlruns")


    # Log other experiment details------------------------------------------------------------

    trainingDataSize = len(data_module.train_dataloader().dataset)
    testingDataSize = len(data_module.val_dataloader().dataset)
    valDataSize = len(data_module.test_dataloader().dataset)

    print(f'Train_size : {trainingDataSize}\n Validation_size: {testingDataSize}\n Test_size: {valDataSize}')

  
    early_stopping_callback = EarlyStopping(
        monitor="Loss_val", patience=5, verbose=True
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/{experiment_name}",
        # every_n_epochs=1,
        filename=f"{data_params['RUN_NAME']}"
        + "-{epoch:02d}-{Loss_val:.2f}",
        #save_top_k=1,
        every_n_epochs=1,
        save_last=True,
        verbose=True,
        #monitor="Loss_val",
        #mode="min",
    )
    # Log other expriment details------------------------------------------------------------

    logger.log_hyperparams(
        data_params.update(
            {
                "train-data-size": len(data_module.train_dataloader().dataset),
                "val-data-size": len(data_module.val_dataloader().dataset),
                "test-data-size": len(data_module.test_dataloader().dataset),
            }
        )
       # data_params
       # | {
       #      "train-data-size": len(data_module.train_dataloader().dataset),
       #      "val-data-size": len(data_module.val_dataloader().dataset),
       #      "test-data-size": len(data_module.test_dataloader().dataset),
       #  }
    )

    # Testing Dataloader --------------------------------------------------------------------

   # test_dataloader(data_module, classifier)

    # Trainer --------------------------------------------------------------------------------
    print("we are in trainer")
    trainer = pl.Trainer(
        # deterministic=True,
        logger=logger,
        callbacks=[
            #early_stopping_callback,
            checkpoint_callback,
            #StochasticWeightAveraging()
        ],

        max_epochs=20,
        # checkpoint_callback=True,
        auto_select_gpus=True,
        # gpus=-1,
        # accelerator = "gpu",
        # devices = -1,
        num_nodes=1,
        #strategy= DDPPlugin(find_unused_parameters=True), #'ddp', #(multiple-gpus, 1 machine)
        # accumulate_grad_batches=16,
        # progress_bar_refresh_rate=500,
        # profiler="simple",
        log_every_n_steps=5,
        num_sanity_val_steps=0,  # use this when AUROC throws error in validation
        #val_check_interval= 1,
        check_val_every_n_epoch=1,
        # resume_from_checkpoint="checkpoints/mimic-cxr-version=29-epoch=02-val_loss=0.08.ckpt",
        # limit_train_batches= data_params["limit_batch"],  # maybe could use this for low resource finetuning
        #fast_dev_run=1,
        # overfit_batches=2,
        # auto_lr_find =True,
        precision=16, 
        # accelerator="gpu"

    )


    checkpoint_path = 'checkpoints/Summarization/last.ckpt'

    if only_test_epoch == False:

        trainer.fit(model=classifier, ckpt_path=checkpoint_path, datamodule=data_module)

        print("*" * 20, f"Training Finished for train size {data_params['train-size']}", "*" * 20)

    print(f"Checkpoint Path: {checkpoint_callback.best_model_path} Score:{checkpoint_callback.best_model_score} ")

 

    print("*" * 20, "Testing", "*" * 20)


   # data_module.process_paper_level = True # Need to make it explicit after sentence-level-training.
   # data_module.setup() # call setup again to re-initialize the Datsets in dataloaders
    #print(f' Process_paper_level ==== {data_module.test_dataset.process_paper_level}')
    pl.Trainer(gpus=-1, logger=logger).test(
        model=classifier,
        ckpt_path='checkpoints/Summarization/last.ckpt',
        dataloaders=data_module.test_dataloader())


if __name__=='__main__':

    data = pd.read_pickle('data/labels.pkl')
    # Create a test and train set.
    trainData, valData, testData = data[data.paper_id<=1000],data[(data.paper_id>1000)&(data.paper_id<=1150)] , data[data.paper_id>1150]
    #trainData, valData, testData = data[data.paper_id<=10],data[(data.paper_id>1000)&(data.paper_id<=1005)] , data[data.paper_id>1400] # Uncomment for testing smaller dataset


    args_dict = {
    "EXP_NAME": 'Summarization',
    "RUN_NAME": 'test',#'ChunkSumm-sentences-conv-500-k50-unfrozen-train-1000-epoch-14-testing-paper-level',
    "BATCH_SIZE":32,
    "BACKBONE": 'bert-base-uncased',
    "N_CLASS": 2,
    "LABELS":['IN','OUT'],
    "freeze_bert": False,
    'train-size': 'full',
    'process_paper_level':False}



        #seeds = [42,34,18,56,0]
        # seeds = [42,34,18,56,0]
        # for seed in [42]:
    pl.seed_everything(42, workers=True)
    train(data_params=args_dict,trainData=trainData,valData=valData,testData=testData,only_test_epoch=False)
