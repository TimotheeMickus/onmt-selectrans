import torch
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Load embeddings.")
    parser.add_argument("vocabpath", type=str, help="path to .vocab.pt file")
    parser.add_argument("embpath", type=str, help="path to embedding file (W2V format, no header)")
    parser.add_argument("--dim", type=int, default=300, help="embeddings dimension")
    parser.add_argument("--checksubfield", action="store_true", help="coerce feature labels in subfield vocab")
    args = parser.parse_args()

    fields = torch.load(args.vocabpath)
    if args.checksubfield :
        subfield_vocab = fields['src'][1][-1].vocab
        if "SLCT" not in subfield_vocab.stoi or "DROP" not in subfield_vocab.stoi:
            print("missing labels in subfield vocab")
            for w in {"SLCT", "DROP"}: #from torchtext
                if w not in subfield_vocab.stoi: 
                    subfield_vocab.itos.append(w)
                    subfield_vocab.stoi[w] = len(subfield_vocab.itos) - 1
            torch.save(fields, args.vocabpath)
        else:
            print("no missing labels in subfield vocab")

    t = fields['src'][0][-1].vocab.stoi
    embs = torch.randn(len(t), args.dim)
    with open(args.embpath, "r") as istr:
        for line in map(str.split, map(str.strip, istr)):
            w, v = line[0], map(float, line[1:])
            if w in t:
                embs[t[w],:] = torch.tensor(list(v))
    out_enc = os.path.join(os.path.dirname(args.vocabpath), "enc.embeddings.pt")
    torch.save(embs, out_enc)

    t = fields['tgt'][0][-1].vocab.stoi
    embs = torch.randn(len(t), args.dim)
    with open(args.embpath, "r") as istr:
        for line in map(str.split, map(str.strip, istr)):
            w, v = line[0], map(float, line[1:])
            if w in t:
                embs[t[w],:] = torch.tensor(list(v))
    out_dec = os.path.join(os.path.dirname(args.vocabpath), "dec.embeddings.pt")
    torch.save(embs, out_dec)
