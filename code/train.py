from argparse import Namespace
from data_generation import generate_tables
from blocking import run_blocking
import random
from summarizer import Summarizer
from DK import ProductDKInjector, GeneralDKInjector
from dataset import DittoDataset
import time
from utils import train
from utils import run_inference

# ---------------------------------------------------------------------------------------------
# TO UPDATE TO THE PROJECT ROOT
base_path_blocking = "D:/Study/ENSIAS/stage_2/ER/ditto/resultat"
# ---------------------------------------------------------------------------------------------


hp = Namespace(
    # Hyperparameters for blocking part
    model_name_blocking="all-MiniLM-L12-v2",
    top_k_blocking=5,
    threshold_blocking=0.95,
    batch_size_blocking=512,

    
    # Paths
    base_path_blocking=base_path_blocking,

    # Input CSVs
    table_reference_csv=f"{base_path_blocking}/data/reference_table.csv",
    table_source_csv=f"{base_path_blocking}/data/source_table.csv",
    ground_truth_csv=f"{base_path_blocking}/data/ground_truth.csv",

    # Ditto-style TXT
    table_reference_txt=f"{base_path_blocking}/input_txt_blocking/reference_table.txt",
    table_source_txt=f"{base_path_blocking}/input_txt_blocking/source_table.txt",

    # Vector files
    table_reference_vec=f"{base_path_blocking}/vectors_blocking/reference_table.txt.mat",
    table_source_vec=f"{base_path_blocking}/vectors_blocking/source_table.txt.mat",

    # Blocking outputs
    output_pairs_csv=f"{base_path_blocking}/blocking/blocking_pairs.csv",
    output_ditto_txt=f"{base_path_blocking}/blocking/blocking_pairs_ditto.txt",

    # Inference output
    output_inference_csv=f"{base_path_blocking}/inference/result.csv",

    dataset_csv_dir=f"{base_path_blocking}/dataset_ditto_csv",
    dataset_txt_dir=f"{base_path_blocking}/dataset_ditto_txt",

    # Logging and task info
    logdir="./logs",
    task="Generated_data",

    # Hyperparameters for training
    batch_size=32,
    lr=3e-5,
    epochs=5,
    save_model=True,
    lm="distilbert",
    size=None,
    alpha_aug=0.8,
    max_len=256,
    da="all",
    summarize=True,
    dk=True,
    fp16=True,
    overwrite=True
)


configs = [{
    "name": "Generated_data",
    "trainset": f"{hp.base_path_blocking}/dataset_ditto_txt/train.txt",
    "validset": f"{hp.base_path_blocking}/dataset_ditto_txt/valid.txt",
    "testset": f"{hp.base_path_blocking}/dataset_ditto_txt/test.txt"
}]


configs = {conf['name'] : conf for conf in configs}
config = configs[hp.task]



def run_full_pipeline(hp, config):
    trainset = config['trainset']
    validset = config['validset']
    testset = config['testset']
    random.seed(42)  # For reproducibility

    #---------------------------------------------------------------------------------------------
    #TO UPDATE IF NEEDED FOR DATA GENERATION
    generate_tables(base_path=hp.base_path_blocking, n_total=2000, match_ratio=0.3)
    #---------------------------------------------------------------------------------------------
    run_blocking(hp)
    if hp.summarize:
        summarizer = Summarizer(config, hp.lm)
        trainset = summarizer.transform_file(trainset, max_len = hp.max_len, overwrite=hp.overwrite)
        testset = summarizer.transform_file(testset, max_len = hp.max_len, overwrite=hp.overwrite)
        validset = summarizer.transform_file(validset, max_len = hp.max_len, overwrite=hp.overwrite)

    if hp.dk is not None:
        if hp.dk == 'product':
            injector = ProductDKInjector(config, hp.dk)
        else:
            injector = GeneralDKInjector(config, hp.dk)
        
        trainset = injector.transform_file(trainset, overwrite=hp.overwrite)
        validset = injector.transform_file(validset, overwrite=hp.overwrite)
        testset = injector.transform_file(testset, overwrite=hp.overwrite)

    train_dataset = DittoDataset(trainset,
                                   lm=hp.lm,
                                   max_len=hp.max_len,
                                   size=hp.size,
                                   da=hp.da)
    valid_dataset = DittoDataset(validset, lm=hp.lm)
    test_dataset = DittoDataset(testset, lm=hp.lm)

    t1 = time.time()
    train(train_dataset, valid_dataset, test_dataset, run_tag="test_run", hp=hp)
    t2 = time.time()

    print(f"Trainig time: {round(t2-t1, 3)} seconds")

run_full_pipeline(hp, config)

if hp.save_model:
    print("====================== INFERENCE 2 EXAMPLES ======================")
    model_path = f"{hp.base_path_blocking}/logs/{hp.task}/model.pt"
    left_str = "COL full_name VAL Michelle Andre COL cin VAL EX717542 COL date_of_birth VAL 1991-02-12 COL place_of_birth VAL Tanger COL cnss_number VAL 36759250 COL email VAL isaacriviere@example.org COL phone VAL +33 3 28 68 25 81 COL address VAL 9, rue Blot COL city VAL Marrakech COL employer_name VAL Menard"
    right_str = "COL full_name VAL Clémence Parent Le Rey COL cin VAL IR469929 COL date_of_birth VAL 1994-06-27 COL place_of_birth VAL Kenitra COL cnss_number VAL 25591412 COL email VAL jweber@example.org COL phone VAL 01 45 59 74 83 COL address VAL rue de Marchand COL city VAL Marrakech COL employer_name VAL Mendès Potier et Fils"
    run_inference(model_path, left_str, right_str, lm=hp.lm, max_len=hp.max_len)
    print("--------------------------------------")
    left_str="COL full_name VAL Audrey Brunet COL cin VAL XE204809 COL date_of_birth VAL 1964-01-04 COL place_of_birth VAL Agadir COL cnss_number VAL 29199351 COL email VAL marcelle91@example.org COL phone VAL +33 7 93 72 16 30 COL address VAL 97, boulevard Colin COL city VAL Marrakech COL employer_name VAL Guillet"
    right_str="COL full_name VAL audrey brunet COL cin VAL XE204809 COL date_of_birth VAL 1964-01-04 COL place_of_birth VAL Agadir COL cnss_number VAL nan COL email VAL marcelle91@example.org COL phone VAL g33 7 93 72 16 30 COL address VAL 97, boulevard Colin COL city VAL Marrakech COL employer_name VAL Guillet"
    run_inference(model_path, left_str, right_str, lm=hp.lm, max_len=hp.max_len)