Barebones translation of [ESMC](https://github.com/evolutionaryscale/esm) to JAX/[equinox](https://docs.kidger.site/equinox/).


```python

from esmj import from_torch
import equinox as eqx
import numpy as np

# load torch model
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
client = ESMC.from_pretrained("esmc_300m").to("cpu")


# demo
prot_seq = "ESCALANTE"

# torch prediction
protein = ESMProtein(sequence=prot_seq)
protein_tensor = client.encode(protein)
torch_output = client.logits(
   protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
)

# translate model to JAX
eqx_model = from_torch(client)
tokens = eqx_model.tokenize(prot_seq)
# jit the model
eqx_model = eqx.filter_jit(eqx_model)
# run it
output = eqx_model(tokens[None]) # add batch dimension

print(np.abs(output.logits - np.array(torch_output.logits.sequence)).max())
# close enough, maybe!
```


This project should be installable using `uv.`
