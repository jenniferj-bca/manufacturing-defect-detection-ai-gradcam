# Dataset setup

Download **Real-life Industrial Dataset of Casting Product** from
[Kaggle](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product).

Either extract the archive so the folders are directly available at:

```text
casting_512x512/casting_512x512/def_front/
casting_512x512/casting_512x512/ok_front/
```

or point `data.root` in `config.yaml` to any directory containing
`def_front/` and `ok_front/`. The loader searches nested directories, so both the
Kaggle archive layout and a flattened layout work. The checked-in image data is
not required for the source code and should not be redistributed.
