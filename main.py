import argparse
import random
import numpy as np
import torch

from model import TST, MetaTST
from dataset import TextStyleDataset, MetaTextStyleDataset
from train import train_kobart, train_meta_kobart
from test import test_kobart, test_meta_kobart
from evaluation import calculate_bleu, caculate_perflexity


def parse_args():
    parser = argparse.ArgumentParser("Meta Text Style Transfer")

    parser.add_argument('--mode', default="train", type=str)
    parser.add_argument('--type', default="meta", type=str)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--weight_decay', default=1e-2, type=float)
    parser.add_argument('--epoch', default=10, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--meta_batch_size', default=4, type=int)
    parser.add_argument('--n_shot', default=5, type=int)
    parser.add_argument('--n_qry', default=30, type=int)
    parser.add_argument('--num_inner_loop', default=5, type=int)
    parser.add_argument('--inner_lr', default=1e-2, type=float)
    parser.add_argument('--max_length', default=64, type=int)
    parser.add_argument('--num_token', default=4, type=int)
    parser.add_argument('--model_name', default="model", type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--ckpt', default=None, type=str)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "train":
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
   
    elif args.mode == "test":
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    if args.mode == "train" and args.type == "meta":
        train_dataset = MetaTextStyleDataset(args)
        model = MetaTST(args)
        model.load_state_dict(torch.load(f"checkpoint/TST.pth", map_location="cpu"), strict=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        train_meta_kobart(model, args, train_dataset, optimizer)
    
    elif args.mode == "train" and args.type == "basic":
        train_dataset = TextStyleDataset(args)
        model = TST(args)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        train_kobart(model, args, train_dataset, optimizer)

    elif args.mode == "test" and args.type == "meta":
        args.num_inner_loop = 10
        test_dataset = MetaTextStyleDataset(args)
        model = MetaTST(args)
        model.load_state_dict(torch.load(f"checkpoint/{args.ckpt}", map_location="cpu"))
        
        test_meta_kobart(model, args, test_dataset)


    elif args.mode == "test" and args.type == "basic":
        test_dataset = TextStyleDataset(args)
        model = TST(args)
        model.load_state_dict(torch.load(f"checkpoint/{args.ckpt}", map_location="cpu"))
        
        test_kobart(model, args, test_dataset)

    elif args.mode == "evaluation" and args.type == "meta":
        args.num_inner_loop = 10
        test_dataset = MetaTextStyleDataset(args)
        model = MetaTST(args)
        model.load_state_dict(torch.load(f"checkpoint/{args.ckpt}", map_location="cpu"))

        calculate_bleu(f"output/{args.ckpt[:-4]}.txt", model.tokenizer)
        caculate_perflexity(model, test_dataset, args.type)

    elif args.mode == "evaluation" and args.type == "basic":
        test_dataset = TextStyleDataset(args)
        model = TST(args)
        model.load_state_dict(torch.load(f"checkpoint/{args.ckpt}", map_location="cpu"))

        calculate_bleu(f"output/{args.ckpt[:-4]}.txt", model.tokenizer)
        caculate_perflexity(model, test_dataset, args.type)
