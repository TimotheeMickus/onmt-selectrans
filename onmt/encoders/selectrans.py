import torch
import torch.nn as nn

from onmt.inputters.inputter import load_old_vocab, old_style_vocab
from onmt.encoders.transformer import TransformerEncoder

SLCT_label = "SLCT"

class SelecTransEncoder(TransformerEncoder):
    """
        Wrapper around a Transformer Encoder.
        Masking elements not labeled with `select_index`.
    """

    def __init__(self, select_idx, *transformer_args):
        super(SelecTransEncoder, self).__init__(*transformer_args)
        self._select_idx = select_idx

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""

        # Retrieve fields
        vocab = torch.load(opt.data + '.vocab.pt')
        if old_style_vocab(vocab):
            fields = load_old_vocab(
                vocab, opt.model_type, dynamic_dict=opt.copy_attn)
        else:
            fields = vocab

        select_idx = fields["src"][-1][-1].vocab.stoi[SLCT_label]

        return cls(
            select_idx,
            opt.enc_layers,
            opt.enc_rnn_size,
            opt.heads,
            opt.transformer_ff,
            opt.dropout,
            embeddings,
            opt.max_relative_positions)

    def forward(self, src, lengths=None):
        """See :func:`TransformerEncoder.forward()`"""

        # Unpack input and features
        true_input, select_labels = torch.unbind(src, dim=-1)
        # Call the transformer
        emb, out, lengths = super(SelecTransEncoder, self).forward( \
            true_input.unsqueeze(-1), lengths)

        # Which output is to be discared
        selector_mask = (select_labels == self._select_idx).unsqueeze(-1)

        # Recomputing output lengths
        memory_lengths = torch.ones(src.size(1), dtype=torch.long).to(lengths.device)

        # TODO: support multiple items selection
        # [1 x B x F]
        select_out = torch.masked_select(out, selector_mask).view(1, \
            src.size(1), out.size(-1)).to(out.device)

        # embeddings are ignored for transformers, so no need to rebatch them
        return emb, select_out, memory_lengths

    def clean_src(self, src):
        # TODO: remove that fix...
        true_input, select_labels = torch.unbind(src, dim=-1)
        selector_mask = (select_labels == self._select_idx)
        select_src = torch.masked_select(true_input, selector_mask)
        # one element, batched, one feature -> [1 x B x 1]
        return select_src.view(1, src.size(1), 1)
