# S2F-GAN
Convert sketches to faces with Generative Adversarial Networks

## Requirements

- Python 3.6+
- [PyTorch](http://pytorch.org/)
- [Pillow](https://python-pillow.org/)
- NVIDIA GPUs with CUDA support

## Usage

```shell
$ python3 main.py --data_path [dataset path] --mode [train|test] --epochs [number of epochs to train]

# Other arguments
$ python3 main.py --help
```

## References

- [WassersteinGAN](https://github.com/martinarjovsky/WassersteinGAN)
  - Paper: https://arxiv.org/abs/1701.07875
- [DiscoGAN-pytorch](https://github.com/carpedm20/DiscoGAN-pytorch)
  - Paper: https://arxiv.org/abs/1703.05192
- [DCGAN](https://arxiv.org/abs/1511.06434)

## License

[GPLv3](https://github.com/mikelei8291/S2F-GAN/blob/master/LICENSE)