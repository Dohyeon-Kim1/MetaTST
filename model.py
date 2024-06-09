import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedTokenizerFast, BartModel
from collections import OrderedDict


class Kobart(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.max_length = args.max_length

        self.kobart = BartModel.from_pretrained("gogamza/kobart-base-v2")
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained("gogamza/kobart-base-v2")
        self.classifier = nn.Linear(768,30000)

    def forward(self, enc_input, dec_input, style):
        enc_input = self.input_transform(enc_input, loc="enc", style=style)
        dec_input = self.input_transform(dec_input, loc="dec")

        output = self.kobart(input_ids=enc_input, decoder_input_ids=dec_input)["last_hidden_state"]
        output = self.classifier(output)
        return output
    
    def generate(self, enc_input, style):
        generated_text = []
        dec_input = torch.LongTensor([[0]]).to(self.kobart.device)
        enc_input = self.input_transform(enc_input, loc="enc", style=style)
        enc_hid = self.kobart.encoder(input_ids=enc_input)[0]
        for _ in range(self.max_length-1):
            output = self.kobart.decoder(input_ids=dec_input, encoder_hidden_states=enc_hid)["last_hidden_state"]
            output = self.classifier(output)
            last_token = output[:,-1,:].argmax(dim=-1)
            dec_input = torch.cat([dec_input, last_token.unsqueeze(0).long()], dim=1)
            generated_text.append(last_token.item())
            if last_token == 1:
                break
        return self.tokenizer.decode(generated_text[:-1])

    def get_loss(self, pred, label):
        pred = pred.permute(0,2,1)
        loss = F.cross_entropy(pred, label, ignore_index=3)
        return loss

    def get_label(self, dec_input):
        return self.input_transform(dec_input, loc="dec")[:,1:]

    def input_transform(self, sent, loc, style=None):
        if loc == "enc":
            sent = list(map(lambda x: x[1] + " 말투로 변환:" + "<s>" + x[0] + "</s>", zip(sent, style)))
            transformed = torch.LongTensor(self.tokenizer(sent, padding="max_length", max_length=self.max_length, truncation=True)["input_ids"]).to(self.kobart.device)
        if loc == "dec":
            sent = list(map(lambda x: "<s>" + x + "</s>", sent))
            transformed = torch.LongTensor(self.tokenizer(sent, padding="max_length", max_length=self.max_length, truncation=True)["input_ids"]).to(self.kobart.device)
        return transformed
    
    def to(self, device):
        super().to(device)


class MetaKobart(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.num_token = args.num_token
        self.max_length = args.max_length
        self.num_inner_loop = args.num_inner_loop
        self.inner_lr = args.inner_lr

        self.kobart = BartModel.from_pretrained("gogamza/kobart-base-v2")
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained("gogamza/kobart-base-v2")
        self.kobart.requires_grad_(False)

        self.style_prompt = nn.Parameter(torch.randn(self.num_token,192))
        self.prompt_mlp = nn.Linear(192,768)
        self.classifier = nn.Linear(768,30000)

    def forward(self, enc_input, dec_input):
        style_prompt = self.prompt_mlp(self.style_prompt)
        enc_input = self.input_transform(enc_input, loc="enc", style_prompt=style_prompt)
        dec_input = self.input_transform(dec_input, loc="dec")

        output = self.kobart(inputs_embeds=enc_input, decoder_input_ids=dec_input)["last_hidden_state"]
        output = self.classifier(output)
        return output

    def functional_forward(self, enc_input, dec_input, weights):
        style_prompt = F.linear(weights["style_prompt"], weight=weights["prompt_mlp.weight"], bias=weights["prompt_mlp.bias"])
        enc_input = self.input_transform(enc_input, loc="enc", style_prompt=style_prompt)
        dec_input = self.input_transform(dec_input, loc="dec")

        output = self.kobart(inputs_embeds=enc_input, decoder_input_ids=dec_input)["last_hidden_state"]
        output = F.linear(output, weights["classifier.weight"], weights["classifier.bias"])
        return output

    def generate(self, enc_input, fast_weights):
        generated_text = []
        dec_input = torch.LongTensor([[0]]).to(self.kobart.device)
        style_prompt = F.linear(fast_weights["style_prompt"], fast_weights["prompt_mlp.weight"], fast_weights["prompt_mlp.bias"]).view(self.num_token,-1)
        enc_input = self.input_transform(enc_input, loc="enc", style_prompt=style_prompt)
        enc_hid = self.kobart.encoder(inputs_embeds=enc_input)[0]
        for _ in range(self.max_length-1):
            output = self.kobart.decoder(input_ids=dec_input, encoder_hidden_states=enc_hid)["last_hidden_state"]
            output = F.linear(output, fast_weights["classifier.weight"], fast_weights["classifier.bias"])
            last_token = output[:,-1,:].argmax(dim=-1)
            dec_input = torch.cat([dec_input, last_token.unsqueeze(0).long()], dim=1)
            generated_text.append(last_token.item())
            if last_token == 1:
                break
        return self.tokenizer.decode(generated_text[:-1])

    def get_loss(self, pred, label):
        pred = pred.permute(0,2,1)
        loss = F.cross_entropy(pred, label, ignore_index=3)
        return loss

    def get_label(self, dec_input):
        return self.input_transform(dec_input, loc="dec")[:,1:]
    
    def input_transform(self, sent, loc, style_prompt=None):
        if loc == "enc":
            sent = list(map(lambda x: "<s>" + x + "</s>", sent))
            sent_embed = self.sent_to_embed(sent)
            prompt = style_prompt.repeat(len(sent),1,1)
            transformed = torch.cat([prompt, sent_embed], dim=1)[:,:self.max_length,:]
        elif loc == "dec":
            sent = list(map(lambda x: "<s>" + x + "</s>", sent))
            transformed = torch.LongTensor(self.tokenizer(sent, padding="max_length", max_length=self.max_length, truncation=True)["input_ids"]).to(self.kobart.device)
        return transformed

    def sent_to_embed(self, sent):
        return self.kobart.shared(torch.LongTensor(self.tokenizer(sent, padding="max_length", max_length=self.max_length, truncation=True)["input_ids"]).to(self.kobart.device))
    
    def inner_loop(self, enc_input, dec_input):
        fast_weights = OrderedDict({"style_prompt": self.style_prompt, "prompt_mlp.weight": self.prompt_mlp.weight, "prompt_mlp.bias": self.prompt_mlp.bias,
                                    "classifier.weight": self.classifier.weight, "classifier.bias": self.classifier.bias})
        label = self.get_label(dec_input)

        for _ in range(self.num_inner_loop):
            pred = self.functional_forward(enc_input, dec_input, fast_weights)
            loss = self.get_loss(pred[:,:-1,:], label)
            gradients = torch.autograd.grad(loss, fast_weights.values(), create_graph=False)
            fast_weights = OrderedDict((name, param - self.inner_lr * grad) for ((name, param), grad) in zip(fast_weights.items(), gradients))
        return fast_weights
    
    def to(self, device):
        super().to(device)
