import unittest
import time
import numpy as np
from typing import List
from tinygrad.nn import optim
from tinygrad.tensor import Tensor, Device
from tinygrad.helpers import getenv
from extra.training import train
from models.convnext import ConvNeXt
from models.efficientnet import EfficientNet
from models.transformer import Transformer
from models.vit import ViT
from models.resnet import ResNet18

BS: int = getenv("BS", 2)

def train_one_step(model,X,Y) -> None:
  params: List[Tensor] = optim.get_parameters(model)
  pcount = sum(np.prod(p.shape) for p in params)
  optimizer = optim.SGD(params, lr=0.001)
  print("stepping %r with %.1fM params bs %d" % (type(model), pcount/1e6, BS))
  st = time.time()
  train(model, X, Y, optimizer, steps=1, BS=BS)
  et = time.time()-st
  print("done in %.2f ms" % (et*1000.))

def check_gc() -> None:
  if Device.DEFAULT == "GPU":
    from extra.introspection import print_objects
    assert print_objects() == 0

class TestTrain(unittest.TestCase):
  def test_convnext(self) -> None:
    model = ConvNeXt(depths=[1], dims=[16])
    X = np.zeros((BS,3,224,224), dtype=np.float32)
    Y = np.zeros((BS), dtype=np.int32)
    train_one_step(model,X,Y)
    check_gc()

  def test_efficientnet(self) -> None:
    model = EfficientNet(0)
    X = np.zeros((BS,3,224,224), dtype=np.float32)
    Y = np.zeros((BS), dtype=np.int32)
    train_one_step(model,X,Y)
    check_gc()

  @unittest.skipIf(Device.DEFAULT == "METAL", "Not working on Metal")
  def test_vit(self) -> None:
    model = ViT()
    X = np.zeros((BS,3,224,224), dtype=np.float32)
    Y = np.zeros((BS,), dtype=np.int32)
    train_one_step(model,X,Y)
    check_gc()

  def test_transformer(self) -> None:
    # this should be small GPT-2, but the param count is wrong
    # (real ff_dim is 768*4)
    model = Transformer(syms=10, maxlen=6, layers=12, embed_dim=768, num_heads=12, ff_dim=768//4)
    X = np.zeros((BS,6), dtype=np.float32)
    Y = np.zeros((BS,6), dtype=np.int32)
    train_one_step(model,X,Y)
    check_gc()

  def test_resnet(self) -> None:
    X = np.zeros((BS, 3, 224, 224), dtype=np.float32)
    Y = np.zeros((BS), dtype=np.int32)
    for resnet_v in [ResNet18]:
      model = resnet_v()
      model.load_from_pretrained()
      train_one_step(model, X, Y)
    check_gc()

  def test_bert(self) -> None:
    # TODO: write this
    pass

if __name__ == '__main__':
  unittest.main()
