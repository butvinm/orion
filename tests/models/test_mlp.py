import time
import math
from pathlib import Path

import torch 
import orion 
import orion.models as models
from orion.core.utils import mae

def get_config_path(yml_str):
    orion_path = Path(__file__).parent.parent
    return str(orion_path / "configs" / f"{yml_str}")

def test_mlp():
    torch.manual_seed(42) # set seed

    # Initialize the Orion scheme and model
    orion.init_scheme(get_config_path("mlp.yml"))
    net = models.MLP()

    inp = torch.randn(1,1,28,28) # MNIST image shape

    # Run cleartext inference
    net.eval()
    out_clear = net(inp)

    # Fit and compile
    orion.fit(net, inp)
    input_level = orion.compile(net)

    # Encode and encrypt the input vector 
    vec_ptxt = orion.encode(inp, input_level)
    vec_ctxt = orion.encrypt(vec_ptxt)
    net.he() # Switch to FHE mode

    # Run FHE inference
    out_ctxt = net(vec_ctxt)

    # Get the FHE results and decrypt + decode.
    out_ptxt = out_ctxt.decrypt()
    out_fhe = out_ptxt.decode()

    dist = mae(out_clear, out_fhe)

    # small tolerable difference depends on parameter set
    assert dist < 0.005 
    