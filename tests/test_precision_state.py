import unittest
import torch
from torch import nn
from weightlab import cast_floating_parameters


class PrecisionStateTests(unittest.TestCase):
    def test_roundtrip_preserves_rope_buffer_alias_and_parameter_tying(self):
        model=nn.Module();model.embedding=nn.Embedding(8,4,dtype=torch.bfloat16)
        model.head=nn.Linear(4,8,bias=False,dtype=torch.bfloat16);model.head.weight=model.embedding.weight
        model.register_buffer('inv_freq',torch.tensor([.0001234567,.000004321],dtype=torch.float32),persistent=False)
        model.original_inv_freq=model.inv_freq
        parameter=model.embedding.weight;before=parameter.detach().clone();buffer=model.inv_freq.clone();alias=model.original_inv_freq
        cast_floating_parameters(model,torch.float32);cast_floating_parameters(model,torch.bfloat16)
        self.assertIs(model.embedding.weight,parameter);self.assertIs(model.head.weight,parameter)
        self.assertTrue(torch.equal(model.embedding.weight,before))
        self.assertIs(model.inv_freq,alias);self.assertIs(model.original_inv_freq,alias)
        self.assertEqual(model.inv_freq.dtype,torch.float32);self.assertTrue(torch.equal(model.inv_freq,buffer))
