import torch
from transformers import PreTrainedTokenizerFast
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm


def calculate_bleu(output_file, tokenizer=None):
    if tokenizer is None:
        tokenizer = PreTrainedTokenizerFast.from_pretrained("gogamza/kobart-base-v2")
    
    f = open(output_file, "r")
    lines = f.readlines()

    refs = list(map(lambda x: x.strip()[8:], filter(lambda x: "answer" in x, lines)))
    gens = list(map(lambda x: x.strip()[10:], filter(lambda x: "generate" in x, lines)))
    
    refs_token = list(map(lambda x: [list(map(lambda y: str(y), x))], tokenizer(refs)["input_ids"]))
    gens_token = list(map(lambda x: list(map(lambda y: str(y), x)), tokenizer(gens)["input_ids"]))
    
    print(f"BLEU-1: {corpus_bleu(refs_token, gens_token, weights=(1,0,0,0))}")
    print(f"BLEU-2: {corpus_bleu(refs_token, gens_token, weights=(0,1,0,0))}")
    print(f"BLEU-3: {corpus_bleu(refs_token, gens_token, weights=(0,0,1,0))}")
    print(f"BLEU-4: {corpus_bleu(refs_token, gens_token, weights=(0,0,0,1))}")


def caculate_perflexity(model, dataset, model_type):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    model.eval()
    all_loss = []
    for i in tqdm(range(len(dataset))):
        if model_type == "basic":
            origin, answer, style = dataset[i]
            with torch.no_grad():
                pred = model([origin], [answer], [style])
                label = model.get_label([answer])
                loss = model.get_loss(pred[:,:-1,:], label)
        elif model_type == "meta":
            spt, qry, style = dataset[i]
            fast_weights = model.inner_loop(spt[0], spt[1])
            with torch.no_grad():
                pred = model.functional_forward(qry[0], qry[1], fast_weights)
                label = model.get_label(qry[1])
                loss = model.get_loss(pred[:,:-1,:], label)
        all_loss.append(loss)
    perflexity = torch.stack(all_loss).mean().item()
    
    print(f"Perflexity: {perflexity}")