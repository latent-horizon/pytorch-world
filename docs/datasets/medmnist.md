# MedMNIST

[MedMNIST v2](https://arxiv.org/abs/2110.14795) is a collection of standardized biomedical image datasets. `MedMNIST` loads one of them as `images` `(N, 3, size, size)` uint8 and `labels` `(N,)`.

> Yang et al., *MedMNIST v2: A large-scale lightweight benchmark for 2D and 3D biomedical image classification*, Scientific Data 2023. [Paper](https://arxiv.org/abs/2110.14795) · [Data](https://zenodo.org/records/10519652) · CC BY 4.0

## Usage

```python
from torch_world.datasets import MedMNIST

blood = MedMNIST("bloodmnist", split="train", size=64)
item = blood[0]               # {"image": (3, 64, 64) uint8, "label": ()}

blood.images, blood.labels    # (N, 3, 64, 64), (N,)
blood.modality                # "blood smear microscopy"
blood.classes                 # ["basophil", "eosinophil", ...]
blood.malignant               # class ids that are malignant (empty here)
blood.info                    # "bloodmnist (train): blood smear microscopy, 8 classes, 11959 images"
```

| argument | |
|---|---|
| `name` | one of the datasets below |
| `split` | `"train"`, `"val"` or `"test"` |
| `size` | `28`, `64`, `128` or `224` |
| `root` | cache directory, `~/.cache/torch_world/medmnist` by default |
| `download` | fetch the file if it is missing; otherwise raise `FileNotFoundError` |

The `.npz` file is downloaded once from Zenodo and checked against its MD5 on every load. It is read without NumPy. Grayscale images are repeated to three channels, so every dataset feeds the same encoder.

## Available datasets

| name | modality | classes | malignant classes | train / val / test |
|---|---|---|---|---|
| `bloodmnist` | blood smear microscopy | 8 cell types | none | 11,959 / 1,712 / 3,421 |
| `breastmnist` | breast ultrasound | malignant, benign lesion | malignant lesion | 546 / 78 / 156 |
| `dermamnist` | dermatoscopy | 7 skin lesions | actinic keratosis, basal cell carcinoma, melanoma | 7,007 / 1,003 / 2,005 |
| `organamnist` | abdominal CT | 11 organs | none | 34,561 / 6,491 / 17,778 |
| `pathmnist` | colon pathology | 9 tissue types | colorectal adenocarcinoma epithelium | 89,996 / 10,004 / 7,180 |

Split sizes are those published with MedMNIST v2.

## Reference

[`MedMNIST`](../api/datasets.md#torch_world.datasets.MedMNIST) in the API reference.
