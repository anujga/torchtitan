import json
import os

import fire
import os
import torch

import torch
from transformers import LlamaForCausalLM, LlamaConfig


def compare_state_dicts(state_dict_1, state_dict_2, diff_threshold=1e-6):
    keys_1 = set(state_dict_1.keys())
    keys_2 = set(state_dict_2.keys())

    # Find keys that are only in one of the models
    only_in_1 = keys_1.difference(keys_2)
    only_in_2 = keys_2.difference(keys_1)

    discrepancies_found = False

    if only_in_1:
        discrepancies_found = True
        print("Keys only in first model:")
        for k in sorted(only_in_1):
            print("  ", k)

    if only_in_2:
        discrepancies_found = True
        print("Keys only in second model:")
        for k in sorted(only_in_2):
            print("  ", k)

    # For keys present in both models, check for numerical differences
    common_keys = keys_1.intersection(keys_2)
    for k in sorted(common_keys):
        tensor_1 = state_dict_1[k]
        tensor_2 = state_dict_2[k]

        # Ensure both are tensors on CPU
        if not torch.allclose(tensor_1, tensor_2, atol=diff_threshold, rtol=0):
            discrepancies_found = True
            # Compute a simple measure of difference
            diff = (tensor_1 - tensor_2).abs().max().item()
            print(f"Values differ for key '{k}'. Max absolute difference: {diff:.8f}")

    if not discrepancies_found:
        print("No discrepancies found between model weights.")

def convert_hf_to_original(hf_config, state_dict):
    new_state_dict = {}

    dim = hf_config.hidden_size
    # def permute(w, n_heads, dim1=dim, dim2=dim):
    #     return w.view(n_heads, dim1 // n_heads // 2, 2, dim2).transpose(1, 2).reshape(dim1, dim2)

    def permute(w_perm, n_heads, dim1=dim, dim2=dim):
        a = dim1 // n_heads // 2
        # Step 1: Undo the final reshape
        w_perm = w_perm.view(n_heads, 2, a, dim2)
        # Step 2: Undo the transpose
        w_perm = w_perm.transpose(1, 2)
        # Step 3: Undo the initial view
        w_orig = w_perm.reshape(dim1, dim2)
        return w_orig

    for k, v in state_dict.items():
        # Example key: "model.embed_tokens.weight"
        if k == "model.embed_tokens.weight":
            new_key = "tok_embeddings.weight"
        elif k == "lm_head.weight":
            new_key = "output.weight"
        elif k == "model.norm.weight":
            new_key = "norm.weight"
        else:
            # For layer parameters
            # pattern: model.layers.{layer_idx}.X
            # Extract layer index
            if k.startswith("model.layers."):
                parts = k.split(".")
                layer_idx = int(parts[2])
                sub_name = parts[3:]

                # Map sub_name from HF to original
                if sub_name[0] == "self_attn":
                    # self_attn.q_proj.weight -> attention.wq.weight
                    # self_attn.k_proj.weight -> attention.wk.weight
                    # self_attn.v_proj.weight -> attention.wv.weight
                    # self_attn.o_proj.weight -> attention.wo.weight
                    attn_map = {
                        "q_proj": "wq",
                        "k_proj": "wk",
                        "v_proj": "wv",
                        "o_proj": "wo"
                    }
                    new_key = f"layers.{layer_idx}.attention.{attn_map[sub_name[1]]}.weight"



                    if sub_name[1] == "q_proj":
                        v = permute(v, hf_config.num_attention_heads)
                    elif sub_name[1] == "k_proj":
                        dims_per_head = hf_config.hidden_size // hf_config.num_attention_heads
                        v = permute(v, hf_config.num_key_value_heads,
                                    dim1=dims_per_head * hf_config.num_key_value_heads)

                elif sub_name[0] == "mlp":
                    # mlp.gate_proj.weight -> feed_forward.w3.weight
                    # mlp.up_proj.weight -> feed_forward.w1.weight
                    # mlp.down_proj.weight -> feed_forward.w2.weight
                    ffn_map = {
                        "gate_proj": "w1",
                        "up_proj": "w3",
                        "down_proj": "w2"
                    }
                    new_key = f"layers.{layer_idx}.feed_forward.{ffn_map[sub_name[1]]}.weight"

                elif sub_name[0] == "input_layernorm" and sub_name[1] == "weight":
                    # input_layernorm -> attention_norm
                    new_key = f"layers.{layer_idx}.attention_norm.weight"
                elif sub_name[0] == "post_attention_layernorm" and sub_name[1] == "weight":
                    # post_attention_layernorm -> ffn_norm
                    new_key = f"layers.{layer_idx}.ffn_norm.weight"
                else:
                    raise ValueError(f"Unexpected sub_name mapping needed: {sub_name}")
            else:
                raise ValueError(f"Key {k} not recognized and not mapped.")

        new_state_dict[new_key] = v
        return new_state_dict



def main(*,
         src_model = "meta-llama/Llama-3.2-1B-Instruct",
         out_model_dir = "output"):

    hf_config = LlamaConfig.from_pretrained(src_model)
    hf_model = LlamaForCausalLM.from_pretrained(src_model)
    state_dict = hf_model.state_dict()


    params = {

        "n_kv_heads": hf_config.n_kv_heads,
        "ffn_dim_multiplier": (hf_config.intermediate_size / hf_config.hidden_size) * 3 / 8,  # 1.5,
        "rope_theta": hf_config.rope_theta,
        "use_scaled_rope": hf_config.rope_scaling is not None,
        "dim": hf_config.hidden_size,
        "multiple_of": 256,  # Usually 256 for LLaMA
        "n_heads": hf_config.num_attention_heads,
        "n_layers": hf_config.num_hidden_layers,
        "norm_eps": hf_config.rms_norm_eps,
        "vocab_size": hf_config.vocab_size,
        "max_seq_len": hf_config.max_position_embeddings
    }

    # Save params.json
    with open(os.path.join(out_model_dir, "params.json"), "w") as f:
        json.dump(params, f, indent=2)

    new_state_dict = convert_hf_to_original(hf_config, state_dict)
    torch.save(new_state_dict, os.path.join(out_model_dir, "consolidated.00.pth"))


if __name__ == "__main__":
    fire.Fire(main)
