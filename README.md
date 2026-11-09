# Rep2Rep

Official implementation of repetition-to-repetition learning for noise-adaptive self-supervised MRI denoising.

## Method

Let `x[0]` and `x[1]` be independent acquisitions of the same image. Rep2Rep predicts the second acquisition from the first and minimizes the complex-valued squared error:

```text
input:  x[0], sigma
target: x[1]
loss:   mean(abs(f(x[0], sigma) - x[1]) ** 2)
```

Because the target noise is independent of the input, the expected training objective is minimized by the underlying signal. The spatial noise map conditions the denoiser on input noise level. This implementation uses CDLNet2D as the reconstruction network; the learning objective is defined by the repetition pairing above.

## Training

`make_pair` selects fixed, distinct repetitions 0 and 1. Supply an iterable of `(repetitions, sigma_map)` batches to `train_epoch`:

```python
import torch
from model import CDLNet2D
from pairing import make_pair
from training import train_epoch

model = CDLNet2D()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
loss = train_epoch(model, optimizer, batches, make_pair)
```

Repetitions use shape `[N, C, H, W]` with complex values. Noise maps use `[1, 1, H, W]` when shared or `[N, 1, H, W]` when repetition-specific. Data loading and preprocessing are supplied by the application.

## Inference

Pass a complex image batch and its real-valued noise map to `denoise`:

```python
from inference import denoise

estimate = denoise(model, image, sigma_map)
```

`image` has shape `[B, C, H, W]`; `sigma_map` has shape `[B, 1, H, W]` and uses the same intensity scale as the image.

## Citations

```bibtex
@article{janjusevicRep2Rep2026,
  author  = {Janju{\v{s}}evi{\'c}, Nikola and Chen, Jingjia and Ginocchio, Luke and Bruno, Mary and Huang, Yuhui and Wang, Yao and Chandarana, Hersh and Feng, Li},
  title   = {Self-Supervised Noise Adaptive {MRI} Denoising via Repetition to Repetition ({Rep2Rep}) Learning},
  journal = {Magnetic Resonance in Medicine},
  volume  = {95},
  number  = {3},
  pages   = {1619--1633},
  year    = {2026},
  doi     = {10.1002/mrm.70155}
}

@article{luoRep2RepShuffle2026,
  author  = {Luo, Renqing and Janju{\v{s}}evi{\'c}, Nikola and Pei, Haoyang and Wang, Yao and Madelin, Guillaume and Feng, Li},
  title   = {{Rep2Rep-Shuffle}: Shuffled Repetition-to-Repetition Learning for Noise-Adaptive Self-Supervised {MRI} Denoising},
  journal = {Magnetic Resonance in Medicine},
  year    = {2026},
  doi     = {10.1002/mrm.70600}
}

@article{janjusevicCDLNet2022,
  author  = {Janju{\v{s}}evi{\'c}, Nikola and Khalilian-Gourtani, Amirhossein and Wang, Yao},
  title   = {{CDLNet}: Noise-Adaptive Convolutional Dictionary Learning Network for Blind Denoising and Demosaicing},
  journal = {IEEE Open Journal of Signal Processing},
  volume  = {3},
  pages   = {196--211},
  year    = {2022},
  doi     = {10.1109/OJSP.2022.3172842}
}
```

## License

PolyForm Noncommercial 1.0.0. See [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
