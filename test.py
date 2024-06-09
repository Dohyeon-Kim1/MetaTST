import os
import torch
from tqdm import tqdm


@torch.no_grad()
def test_kobart(model, args, dataset):
    if not os.path.exists("output"):
        os.mkdir("output")
    
    f = open(f"output/{args.ckpt[:-4]}.txt", "w")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    model.eval()
    for i in tqdm(range(len(dataset))):
        origin, answer, style = dataset[i]
        
        content = f"""
        
        origin: {origin}, style: {style}
        generate: {model.generate([origin], [style])}
        answer: {answer}

        """

        f.write(content)
    f.close()


def test_meta_kobart(model, args, dataset):
    if not os.path.exists("checkpoint"):
        os.mkdir("checkpoint")
    
    f = open(f"output/{args.ckpt[:-4]}.txt", "w")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    model.eval()
    for i in tqdm(range(len(dataset))):
        spt, qry, style = dataset[i]
        fast_weights = model.inner_loop(spt[0], spt[1])

        with torch.no_grad():
            content = f"""
            
            origin: {qry[0][0]}, style: {style}
            generate: {model.generate([qry[0][0]], fast_weights)}
            answer: {qry[1][0]}

            """
        f.write(content)
    f.close()