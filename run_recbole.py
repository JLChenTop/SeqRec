
import argparse

# from recbole.quick_start import run
from recbole.quick_start import run
from model.gru4rec_test import GRU4Rec_test
from NewModel import NewModel



config_dict={
    "gpu_id":0,
    # "epochs":1,
    "train_batch_size":4096, 
    'loss_type': 'CE'   ,              # (str) The type of loss function. Range in ['BPR', 'CE'].
    'train_neg_sample_args': None
}
# model = "GRU4Rec"
model = "GRU4Rec_test"
model = "NewModel"

dataset = "yoochoose-clicks-merged"
dataset = "diginetica-merged"

dataset = "amazon-books-18"
dataset = "amazon-all-beauty-18"

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

    run(
        args.model,
        args.dataset,
        config_file_list=config_file_list,
        config_dict=args.config_dict,
        nproc=args.nproc,
        world_size=args.world_size,
        ip=args.ip,
        port=args.port,
        group_offset=args.group_offset
    )