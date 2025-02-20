import torch
import quant
import torch.nn.functional as F

@torch.jit.script
def apply_rotary_pos_emb(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
    # x: [sq, b, np, hn]
    sq, b, np, hn = x.size(0), x.size(1), x.size(2), x.size(3)
    rot_dim = rope_cache.shape[-2] * 2
    x, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    # truncate to support variable sizes
    rope_cache = rope_cache[:sq]
    xshaped = x.reshape(sq, -1, np, rot_dim // 2, 2)
    rope_cache = rope_cache.view(sq, -1, 1, xshaped.size(3), 2)

    x_out2 = torch.stack(
        [
            xshaped[..., 0] * rope_cache[..., 0] -
            xshaped[..., 1] * rope_cache[..., 1],
            xshaped[..., 1] * rope_cache[..., 0] +
            xshaped[..., 0] * rope_cache[..., 1],
        ],
        -1,
    )

    x_out2 = x_out2.flatten(3)
    output = torch.cat((x_out2, x_pass), dim=-1)
    return output


class RotaryPosEmb(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, rope_cache):
        return apply_rotary_pos_emb(input, rope_cache)

    @staticmethod
    def symbolic(g, input, rope_cache):
        return g.op("custom::RotaryPosEmb", input, rope_cache)


class Attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query_layer, key_layer, value_layer):
        return torch.nn.functional.scaled_dot_product_attention(
                query_layer.to(torch.float32), key_layer.to(torch.float32), value_layer.to(torch.float32), is_causal=True).to(torch.float16)

    @staticmethod
    def symbolic(g, query_layer, key_layer, value_layer):
        return g.op("custom::Attention", query_layer, key_layer, value_layer)


class RMSNorm(torch.nn.Module): 
    def __init__(self, normalized_shape, eps=1e-5, device=None, dtype=None, **kwargs):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(
            normalized_shape, device=device, dtype=dtype))
        self.eps = eps

    def forward_old(self, hidden_states: torch.Tensor):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(
            2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return (self.weight * hidden_states).to(input_dtype)

    def forward(self, hidden_states):
        return LayerNorm.apply(hidden_states, self.weight, self.eps)


class LayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states, weight, eps):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(
            2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + eps)
        return (weight * hidden_states).to(input_dtype)

    @staticmethod
    def symbolic(g, hidden_states, weight, eps):
        attrs = {"eps_f" : eps}
        return g.op("custom::LayerNorm", hidden_states, weight, **attrs)


class Silu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, data):
        return F.silu(data)

    @staticmethod
    def symbolic(g, data):
        return g.op("custom::Silu", data)


class GLM2Block(torch.nn.Module):

    def __init__(self):
        super(GLM2Block, self).__init__()
        self.input_layernorm = RMSNorm(4096, dtype=torch.float16)
        self.post_atten_layernorm = RMSNorm(4096, dtype=torch.float16)
        self.pos_emb = RotaryPosEmb.apply
        self.atten = Attention.apply
        self.silu = Silu.apply
        self.query_key_value = quant.QuantLinear(bits=4, groupsize=128, infeatures=4096, outfeatures=4608, bias=True)
        self.dense = quant.QuantLinear(bits=4, groupsize=128, infeatures=4096, outfeatures=4096, bias=False)
        self.dense_h_to_4h = quant.QuantLinear(bits=4, groupsize=128, infeatures=4096, outfeatures=27392, bias=False)
        self.dense_4h_to_h = quant.QuantLinear(bits=4, groupsize=128, infeatures=13696, outfeatures=4096, bias=False)

    def forward(self, inputs):
        hidden_states, rotary_pos_emb = inputs
        layernorm_output = self.input_layernorm(hidden_states)
        mixed_x_layer = self.query_key_value(layernorm_output)
        (query_layer, key_layer, value_layer) = mixed_x_layer.split(
            [32 * 128, 2 * 128, 2 * 128, ], dim=-1, )

        query_layer = query_layer.view(query_layer.size()[:-1] + (32, 128))
        key_layer = key_layer.view(key_layer.size()[:-1] + (2, 128))
        value_layer = value_layer.view(value_layer.size()[:-1] + (2, 128))

        query_layer = self.pos_emb(query_layer, rotary_pos_emb)
        key_layer = self.pos_emb(key_layer, rotary_pos_emb)

        key_layer = key_layer.unsqueeze(-2)
        key_layer = key_layer.expand(-1, -1, -1, 32 // 2, -1)
        key_layer = key_layer.contiguous().view(
            key_layer.size()[:2] + (32, 128))
        
        value_layer = value_layer.unsqueeze(-2)
        value_layer = value_layer.expand(-1, -1, -1, 32 // 2, -1)
        value_layer = value_layer.contiguous().view(
            value_layer.size()[:2] + (32, 128))
        
        query_layer, key_layer, value_layer = [
            k.permute(1, 2, 0, 3) for k in [query_layer, key_layer, value_layer]]

        context_layer = self.atten(query_layer, key_layer, value_layer)

        context_layer = context_layer.permute(2, 0, 1, 3)
        new_context_layer_shape = context_layer.size()[:-2] + (128 * 32, )
        context_layer = context_layer.reshape(*new_context_layer_shape)

        attention_output = self.dense(context_layer)

        residual = hidden_states
        layernorm_input = attention_output
        layernorm_input = residual + layernorm_input

        layernorm_output_att = self.post_atten_layernorm(
            layernorm_input)
        
        intermediate_parallel = self.dense_h_to_4h(layernorm_output_att)
        imps = torch.chunk(intermediate_parallel, 2, dim=-1)
        Ele_output = self.silu(imps[0])
        intermediate_parallel = Ele_output * imps[1]
        mlp_output = self.dense_4h_to_h(intermediate_parallel)
        residual = layernorm_input
        hidden_states = mlp_output
        hidden_states = residual + hidden_states
        return hidden_states


class MyGML2(torch.nn.Module):
    def __init__(self, device='cuda'):
        super(MyGML2, self).__init__()
        self.device = device
        self.seq_length = 32768
        self.word_embeddings = torch.nn.Embedding(
            65024, 4096, dtype=torch.float16, device=device)

        self.dim = 64
        self.input_layernorm = torch.nn.ModuleList()
        self.post_attention_layernorm = torch.nn.ModuleList()
        self.query_key_value = torch.nn.ModuleList()
        self.dense = torch.nn.ModuleList()
        self.dense_h_to_4h = torch.nn.ModuleList()
        self.dense_4h_to_h = torch.nn.ModuleList()

        for i in range(28):
            self.input_layernorm.append(
                RMSNorm(4096, device=device, dtype=torch.float16))
            self.post_attention_layernorm.append(
                RMSNorm(4096, device=device, dtype=torch.float16))

            self.query_key_value.append(quant.QuantLinear(
                bits=4, groupsize=128, infeatures=4096, outfeatures=4608, bias=True).to(device))
            self.dense.append(quant.QuantLinear(
                bits=4, groupsize=128, infeatures=4096, outfeatures=4096, bias=False).to(device))

            self.dense_h_to_4h.append(quant.QuantLinear(
                bits=4, groupsize=128, infeatures=4096, outfeatures=27392, bias=False).to(device))
            self.dense_4h_to_h.append(quant.QuantLinear(
                bits=4, groupsize=128, infeatures=13696, outfeatures=4096, bias=False).to(device))

        self.final_layernorm = RMSNorm(
            4096, device=device, dtype=torch.float16)
        self.output_layer = torch.nn.Linear(
            4096, 65024, bias=False, dtype=torch.float16)

    def swiglu(self, x):
        x = torch.chunk(x, 2, dim=-1)
        return F.silu(x[0]) * x[1]

    def forward(self, inputs):
        embeddings = self.word_embeddings(inputs["input_ids"])
        embeddings = embeddings.transpose(0, 1).contiguous()
        embeddings.to(torch.float16)

        theta = 1.0 / (10000 ** (torch.arange(0, self.dim, 2,
                                              dtype=torch.float16, device=self.device) / self.dim))

        seq_idx = torch.arange(
            self.seq_length, dtype=torch.float16, device=self.device)
        idx_theta = torch.outer(seq_idx, theta).float()
        rotary_pos_emb = torch.stack(
            [torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1).half()
        rotary_pos_emb = rotary_pos_emb[inputs["position_ids"]]
        rotary_pos_emb = rotary_pos_emb.transpose(0, 1).contiguous()

        hidden_states = embeddings

        for i in range(28):
            torch.cuda.empty_cache()
            layernorm_output = self.input_layernorm[i](hidden_states)
            mixed_x_layer = self.query_key_value[i](layernorm_output)
            (query_layer, key_layer, value_layer) = mixed_x_layer.split(
                [32 * 128, 2 * 128, 2 * 128, ], dim=-1, )

            query_layer = query_layer.view(query_layer.size()[:-1] + (32, 128))
            key_layer = key_layer.view(key_layer.size()[:-1] + (2, 128))
            value_layer = value_layer.view(value_layer.size()[:-1] + (2, 128))

            query_layer = apply_rotary_pos_emb(query_layer, rotary_pos_emb)
            key_layer = apply_rotary_pos_emb(key_layer, rotary_pos_emb)

            key_layer = key_layer.unsqueeze(-2)
            key_layer = key_layer.expand(-1, -1, -1, 32 // 2, -1)
            key_layer = key_layer.contiguous().view(
                key_layer.size()[:2] + (32, 128))
            
            value_layer = value_layer.unsqueeze(-2)
            value_layer = value_layer.expand(-1, -1, -1, 32 // 2, -1)
            value_layer = value_layer.contiguous().view(
                value_layer.size()[:2] + (32, 128))
            
            query_layer, key_layer, value_layer = [
                k.permute(1, 2, 0, 3) for k in [query_layer, key_layer, value_layer]]

            context_layer = torch.nn.functional.scaled_dot_product_attention(
                query_layer, key_layer, value_layer, is_causal=True)

            context_layer = context_layer.permute(2, 0, 1, 3)
            new_context_layer_shape = context_layer.size()[:-2] + (128 * 32, )
            context_layer = context_layer.reshape(*new_context_layer_shape)

            attention_output = self.dense[i](context_layer)

            residual = hidden_states
            layernorm_input = torch.nn.functional.dropout(
                attention_output, p=0.0, training=False)
            layernorm_input = residual + layernorm_input

            layernorm_output_att = self.post_attention_layernorm[i](
                layernorm_input)
            
            intermediate_parallel = self.dense_h_to_4h[i](layernorm_output_att)
            imps = torch.chunk(intermediate_parallel, 2, dim=-1)
            Ele_output = F.silu(imps[0])
            intermediate_parallel = Ele_output * imps[1]
            mlp_output = self.dense_4h_to_h[i](intermediate_parallel)
            residual = layernorm_input
            hidden_states = torch.nn.functional.dropout(
                mlp_output, p=0.0, training=False)
            hidden_states = residual + hidden_states

        hidden_states = self.final_layernorm(hidden_states)
        lm_logits = self.output_layer(hidden_states)
        lm_logits = lm_logits.transpose(0, 1).contiguous()
        return lm_logits


model = GLM2Block()
# test = model([torch.randn(19, 1, 4096), torch.randn(64, 128)])
torch.onnx.export(
                    model, 
                    [
                        torch.randn(19, 1, 4096).to(torch.float16), 
                        torch.randn(64, 128).to(torch.float16),
                    ], 
                    "./test/glm2_block.onnx", 
                    input_names=["input1", "input2"],
                    output_names=["output"]
)