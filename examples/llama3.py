from pathlib import Path
from typing import List
import json, argparse
import tiktoken
from tiktoken.load import load_tiktoken_bpe
from tqdm import tqdm
from extra.models.llama import Transformer, convert_from_huggingface, fix_bf16
from tinygrad.nn.state import safe_load, torch_load, load_state_dict
from tinygrad import Tensor, dtypes, nn, Context, Device, GlobalCounters

class Tokenizer:
  pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
  def __init__(self, model_path: str):
    mergeable_ranks = load_tiktoken_bpe(model_path)
    special_tokens = [
      "<|begin_of_text|>",
      "<|end_of_text|>",
      "<|reserved_special_token_0|>",
      "<|reserved_special_token_1|>",
      "<|reserved_special_token_2|>",
      "<|reserved_special_token_3|>",
      "<|start_header_id|>",
      "<|end_header_id|>",
      "<|reserved_special_token_4|>",
      "<|eot_id|>",
    ] + [
      f"<|reserved_special_token_{i}|>"
      for i in range(5, 256 - 5)
    ]
    self.special_tokens = {token: len(mergeable_ranks) + i for i, token in enumerate(special_tokens)}

    self.model = tiktoken.Encoding(name=model_path, pat_str=self.pat_str, mergeable_ranks=mergeable_ranks, special_tokens=self.special_tokens)

  @property
  def bos_id(self): return self.special_tokens["<|begin_of_text|>"]
  @property
  def stop_tokens(self): return {self.special_tokens["<|end_of_text|>"], self.special_tokens["<|eot_id|>"]}

  def decode(self, toks): return self.model.decode(toks)
  def encode(self, text, allow_special=False): return self.model.encode(text, allowed_special="all" if allow_special else set())

# **** helper functions ****
def concat_weights(models, device=None):
  def convert(name) -> Tensor:
    disk_tensors: List[Tensor] = [model[name] for model in models]
    if len(disk_tensors) == 1 or len(disk_tensors[0].shape) == 1:
      return disk_tensors[0].to(device=device)
    axis = 1 if name.endswith(".attention.wo.weight") or name.endswith(".feed_forward.w2.weight") else 0
    lazy_tensors = [data.to(device=device) for data in disk_tensors]
    return lazy_tensors[0].cat(*lazy_tensors[1:], dim=axis)
  return {name: convert(name) for name in {name: None for model in models for name in model}}

def load(fn:str):
  if fn.endswith('.index.json'):
    with open(fn) as fp: weight_map = json.load(fp)['weight_map']
    parts = {n: load(str(Path(fn).parent / Path(n).name)) for n in set(weight_map.values())}
    return {k: parts[n][k] for k, n in weight_map.items()}
  elif fn.endswith(".safetensors"):
    return safe_load(fn)
  else:
    return torch_load(fn)

# **** quantized linears ****
class Int8Linear:
  def __init__(self, in_features, out_features, bias=False):
    assert bias == False
    self.weight = Tensor.ones(out_features, in_features, dtype=dtypes.int8)
    self.scale = Tensor.ones(out_features, dtype=dtypes.half)

  def __call__(self, x):
    return x.dot(self.weight.cast(dtype=dtypes.half).T*self.scale)

  @staticmethod
  def quantize(tensors, device):
    new_tensors = {}
    for name,v in tensors.items():
      if "feed_forward" in name or "attention.w" in name:
        assert "weight" in name, name
        scale = v.abs().max(axis=1) / 127.0
        int8_weight = (v.T/scale).T.cast(dtype=dtypes.int8)
        new_tensors[name] = int8_weight
        new_tensors[name.replace('weight', 'scale')] = scale
        if isinstance(device, tuple):
          new_tensors[name].shard_(device, axis=-1)
          new_tensors[name.replace('weight', 'scale')].shard_(device, axis=None)
      else:
        new_tensors[name] = v
    return new_tensors

def NF4Linear(block_size):
  _CODE = [
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
  ]
  CODE = Tensor.stack(*[Tensor(c) for c in _CODE])
  class _NF4Linear:
    def __init__(self, in_features, out_features, bias=False):
      assert not bias, "bias not supported"
      self.in_features, self.out_features = in_features, out_features
      self.weight = Tensor.empty(int(out_features * in_features / 2), dtype=dtypes.uint8)
      self.scale = Tensor.empty(int(out_features * in_features / block_size), 1, dtype=dtypes.float16)

    def __call__(self, x: Tensor) -> Tensor:
      high_bits = self.weight
      low_bits = (self.weight * 2 ** 4).contiguous()
      unpacked = Tensor.stack(high_bits, low_bits, dim=-1).div(2 ** 4, upcast=False)
      unscaled = CODE[unpacked].to(x.device).reshape(-1, block_size) * self.scale
      return x.linear(unscaled.reshape(self.out_features, self.in_features).T)

    @staticmethod
    def quantize(state_dict: dict[str, Tensor], device) -> dict[str, Tensor]:
      new_state_dict = {}
      for k, v in state_dict.items():
        if "feed_forward" in k or "attention.w" in k:
          grouped = v.reshape(-1, block_size)
          scale = (grouped.abs().max(axis=1, keepdim=True))
          coded = ((grouped / scale).unsqueeze(-1) - CODE.to(v.device)).abs().argmin(axis=-1).cast(dtypes.uint8).flatten()
          new_state_dict[k] = coded[::2] * 2 ** 4 + coded[1::2]
          new_state_dict[k.replace(".weight", ".scale")] = scale.cast(dtypes.float16)
          if isinstance(device, tuple):
            new_state_dict[k].shard_(device, axis=-1)
            new_state_dict[k.replace('weight', 'scale')].shard_(device, axis=None)
        else:
          new_state_dict[k] = v
      return new_state_dict
  return _NF4Linear

MODEL_PARAMS = {
  "8B": {
    "args": {"dim": 4096, "n_heads": 32, "n_kv_heads": 8, "n_layers": 32, "norm_eps": 1e-5, "rope_theta": 500000, "vocab_size": 128256, "hidden_dim": 14336},
    "files": 1
  },
  "70B": {
    "args": {"dim": 8192, "n_heads": 64, "n_kv_heads": 8, "n_layers": 80, "norm_eps": 1e-5, "rope_theta": 500000, "vocab_size": 128256,  "hidden_dim": 28672},
    "files": 8
  }
}
def build_transformer(model_path: Path, model_size="8B", quantize=None, device=None):
  # build model
  if quantize == "int8": linear = Int8Linear
  elif quantize == "nf4": linear = NF4Linear(64)
  else: linear = nn.Linear
  with Context(THREEFRY=0):
    model = Transformer(**MODEL_PARAMS[model_size]["args"], linear=linear, max_context=8192, jit=True)

  # load weights
  if model_path.is_dir():
    weights = concat_weights([load(str(model_path / f"consolidated.{i:02d}.pth")) for i in range(MODEL_PARAMS[model_size]["files"])], device[0] if isinstance(device, tuple) else device)
  else:
    weights = load(str(model_path))
  if "model.embed_tokens.weight" in weights:
    weights = convert_from_huggingface(weights, model, MODEL_PARAMS[model_size]["args"]["n_heads"], MODEL_PARAMS[model_size]["args"]["n_kv_heads"])
  weights = fix_bf16(weights)

  with Context(BEAM=0):
    # quantize
    if quantize is not None:
      weights = linear.quantize(weights, device)
      for _,v in weights.items(): v.realize()

    # shard
    if isinstance(device, tuple):
      for k,v in nn.state.get_state_dict(model).items():
        if 'scale' in k: v.shard_(device, axis=None)  # from quantized
        elif '.attention.' in k: v.shard_(device, axis=-1)
        elif '.feed_forward.' in k: v.shard_(device, axis=-1)
        elif 'tok_embeddings.weight' in k: v.shard_(device, axis=0)
        elif 'output.weight' in k: v.shard_(device, axis=0)
        else: v.shard_(device, axis=None)

    # replace weights in model
    load_state_dict(model, weights, strict=False, consume=True)
  return model

# default settings
TEMPERATURE = 0.85
TOP_K = 25
TOP_P = 0.9
ALPHA_F = 1.1
ALPHA_P = 0.0

def prefill(model, toks, start_pos=0):
  for tok in tqdm(toks):
    GlobalCounters.reset()
    model(Tensor([[tok]], device=device), start_pos, TEMPERATURE, TOP_K, TOP_P, ALPHA_F, ALPHA_P).realize()
    start_pos += 1
  return start_pos

if __name__ == "__main__":
  with Tensor.no_grad():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--size", choices=["8B", "70B"], default="8B")
    parser.add_argument("--shard", type=int, default=1)
    parser.add_argument("--quantize", choices=["int8", "nf4"])
    parser.add_argument("--api", action="store_true")
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()

    if args.seed is not None: Tensor.manual_seed(args.seed)
    print(f"seed = {Tensor._seed}")

    tokenizer = Tokenizer(str((args.model if args.model.is_dir() else args.model.parent) / "tokenizer.model"))
    def encode_role(role: str):
      return [tokenizer.special_tokens["<|start_header_id|>"]] + tokenizer.encode(role) + [tokenizer.special_tokens["<|end_header_id|>"]] + tokenizer.encode("\n\n")
    def encode_message(role: str, content: str):
      return encode_role(role) + tokenizer.encode(content.strip()) + [tokenizer.special_tokens["<|eot_id|>"]]

    device = tuple(f"{Device.DEFAULT}:{i}" for i in range(args.shard)) if args.shard > 1 else Device.DEFAULT
    model = build_transformer(args.model, model_size=args.size, quantize=args.quantize, device=device)

    if args.api:
      from bottle import Bottle, request, response, HTTPResponse, abort
      app = Bottle()

      cors_headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
        "Access-Control-Allow-Headers": "Origin, Accept, Content-Type, X-Requested-With, X-CSRF-Token, Authorization",
        "Access-Control-Allow-Credentials": "true",
      }
      @app.hook("before_request")
      def handle_options():
        if request.method == "OPTIONS": raise HTTPResponse(headers=cors_headers)
      @app.hook("after_request")
      def enable_cors():
        for key, value in cors_headers.items(): response.set_header(key, value)

      @app.get("/v1/models")
      def models():
        return json.dumps([str(args.model)])

      @app.post("/v1/internal/token-count")
      def token_count():
        rjson = json.loads(request.body.read())
        return json.dumps(len(tokenizer.encode(rjson.get("text", ""))))
      @app.post("/v1/token/encode")
      def token_encode():
        rjson = json.loads(request.body.read())
        return json.dumps(tokenizer.encode(rjson.get("text", "")))

      @app.post("/v1/completions")
      def completions():
        rjson = json.loads(request.body.read())

        # check if we are streaming
        if rjson.get("stream", False):
          response.content_type = "text/event-stream"
          response.set_header("Cache-Control", "no-cache")
        else: abort(400, "streaming required")

        toks = [tokenizer.bos_id] + tokenizer.encode(rjson.get("prompt", ""), allow_special=True)

        start_pos = prefill(model, toks)
        last_tok = toks[-1]
        while True:
          GlobalCounters.reset()
          tok = model(Tensor([[last_tok]], device=device), start_pos, TEMPERATURE, TOP_K, TOP_P, ALPHA_F, ALPHA_P).item()
          start_pos += 1
          last_tok = tok
          if tok in tokenizer.stop_tokens: break

          res = {
            "choices": [{
              "text": tokenizer.decode([tok]),
            }]
          }
          yield f"data: {json.dumps(res)}\n\n"

      @app.post("/v1/chat/completions")
      def chat_completions():
        rjson = json.loads(request.body.read())
        if "messages" not in rjson: abort(400, "messages required")

        # check if we are streaming
        if rjson.get("stream", False):
          response.content_type = "text/event-stream"
          response.set_header("Cache-Control", "no-cache")
        else: abort(400, "streaming required")

        toks = [tokenizer.bos_id]
        for message in rjson["messages"]:
          toks += encode_message(message["role"], message["content"])
        # ensure that the last message was a user message
        if message["role"] != "user": abort(400, "last message must be a user message")
        toks += encode_role("assistant")

        start_pos = prefill(model, toks[:-1])
        last_tok = toks[-1]
        while True:
          GlobalCounters.reset()
          tok = model(Tensor([[last_tok]], device=device), start_pos, TEMPERATURE, TOP_K, TOP_P, ALPHA_F, ALPHA_P).item()
          start_pos += 1
          last_tok = tok
          if tok in tokenizer.stop_tokens: break

          res = {
            "choices": [{
              "delta": {
                "role": "assistant",
                "content": tokenizer.decode([tok]),
              }
            }]
          }
          yield f"data: {json.dumps(res)}\n\n"

      app.run(host="0.0.0.0", port=7776, debug=True)
    else:
      prompt = [tokenizer.bos_id] + encode_message("system", "You are an *emotive* assistant.")

      start_pos = prefill(model, prompt)
      while True:
        toks = encode_message("user", input("Q: ")) + encode_role("assistant")

        start_pos = prefill(model, toks, start_pos=start_pos)
        last_tok = toks[-1]
        while True:
          GlobalCounters.reset()
          tok = model(Tensor([[last_tok]], device=device), start_pos, TEMPERATURE, TOP_K, TOP_P, ALPHA_F, ALPHA_P).item()
          start_pos += 1
          last_tok = tok
          if tok in tokenizer.stop_tokens: break
          print(tokenizer.decode([tok]), end="", flush=True)
        print(flush=True)
