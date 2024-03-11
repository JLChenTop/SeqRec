from logging import getLogger
from recbole.utils import init_logger, init_seed,get_model
from recbole.trainer import Trainer
from NewModel import NewModel
from model.gru4rec_test import GRU4Rec_test
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
 
import argparse

config_dict={
    "gpu_id":0,
    'embedding_size': 64,
    # "epochs":1,
    "train_batch_size":4096, 
    'dropout_prob' : 0.3,
    'loss_type': 'CE'   ,              # (str) The type of loss function. Range in ['BPR', 'CE'].
    'train_neg_sample_args': None
}
model = "GRU4Rec" 
# model = NewModel
# model = GRU4Rec_test


dataset = "yoochoose-clicks-merged"
dataset = "diginetica-merged"

dataset = "amazon-books-18"
dataset = "amazon-all-beauty-18"
# dataset = "ml-100k"

# device = torch.device(f'cuda:{config_dict["gpu_id"]}' if torch.cuda.is_available() else 'cpu')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default=model, help="name of models")
    parser.add_argument("--dataset", "-d", type=str, default=dataset, help="name of datasets")
    parser.add_argument("--config_files", type=str, default='conf/config.yaml', help="config files")
    parser.add_argument("--config_dict", type=str, default=config_dict, help="config files")
    parser.add_argument("--nproc", type=int, default=1, help="the number of process in this group")
    parser.add_argument( "--ip", type=str, default="localhost", help="the ip of master node")
    parser.add_argument("--port", type=str, default="5678", help="the port of master node")
    parser.add_argument("--world_size", type=int, default=-1, help="total number of jobs")
    parser.add_argument("--group_offset",type=int, default=0, help="the global rank offset of this group",)

    args, _ = parser.parse_known_args()

    config_file_list = (args.config_files.strip().split(" ") if args.config_files else None)


    config = Config(
        model=args.model,
        dataset=args.dataset,
        config_file_list=config_file_list,
        config_dict=args.config_dict,
    )
    # config = Config(model=NewModel, dataset=args.dataset,config_file_list=config_file_list,config_dict=args.config_dict,)
    # print("**************************",args.model)
    # config = Config(model=model, dataset=args.dataset)
    init_seed(config['seed'], config['reproducibility'])

    # logger initialization
    init_logger(config)
    logger = getLogger()

    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    if isinstance(args.model, str):
        model = get_model(args.model)(config, train_data.dataset).to(config['device'])
    else:
        model = args.model(config, train_data.dataset).to(config['device'])
    logger.info(model)

    # trainer loading and initialization
    trainer = Trainer(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)

    # model evaluation
    test_result = trainer.evaluate(test_data)

    logger.info('best valid result: {}'.format(best_valid_result))
    logger.info('test result: {}'.format(test_result))