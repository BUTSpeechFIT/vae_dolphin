# Integration of variational autoencoder and spatial clustering for adaptive multi-channel neural speech separation

This repository is the official implementation of [Integration of variational autoencoder and spatial clustering for adaptive multi-channel neural speech separation](https://arxiv.org/abs/2011.11984).

## Requirements

To install requirements:
```
pip install -r requirements.txt
pip install git+git://github.com/fgnt/pb_bss.git@0299369b257b761ffffd9962f5f2f9fa14dd43c4
```
The code was tested with Python 3.6.11.

## Data
We use the spatialized version of WSJ0-2mix with additional white noise. To generate the spatialized WSJ0-2mix, please follow the [original scripts](http://www.merl.com/demos/deep-clustering). To add the white noise, you can run:
```
python tools/add_white_noise.py <path-to-WSJ0-2mix-spat> <output-directory>
```
- `<path-to-WSJ0-2mix-spat>` is path to the spatialized version. It should contain sub-directories `wsj0-mix/2speakers_reverb/wav8k/min/tt/mix`.
- `<output-directory>` will as a result contain 5 sub-directories with different SNR ranges `wsj0-mix_noise_<low-SNR>_<high-SNR>`

The generation should take about 30 minutes (but it depends highly on the speed of I/O) and take about 11G of memory.

The scripts introduced below use JSON files describing the data. To create the necessary lists, run
```
python tools/create_jsons.py <path-to-WSJ0-2mix> <path-to-WSJ0-2mix-spat> <path-to-WSJ0-2mix-with-noise>
```
- `<path-to-WSJ0-2mix>` is path to the original (not-spatialized) version of WSJ0-2mix. It should contain sub-directories `data/2speakers/wav8k/min/tt`
- `<path-to-WSJ0-2mix-spat>` is path to the spatialized version (same as above).
- `<path-to-WSJ0-2mix-with-noise>` is the output directory of `add_white_noise.py`

JSON files will be generated into `data` folder. You can see examples of the JSONs in this repository in `data`.

## Pre-trained models
Folder `pretrained_models` contains trained VAE, GMM and mean-std normalization, which were used for experiments in the paper. To use these models in the experiments below, run following lines to copy them to proper location:
```
mkdir -p exp/gmm/models; cp pretrained_models/GMM.h5 exp/gmm/models
mkdir -p exp/vae/models; cp pretrained_models/vae exp/vae/models/model.100
mkdir -p exp/noise_model; cp -r pretrained_models/noise_models exp/noise_model/models
cp pretrained_models/vae_config exp/vae/models/vae_config
cp pretrained_models/tr_meanstd data/wsj0-mix_spat/tr_meanstd
```
Alternatively, you can just change paths in configuration of the experiment in `run.py`.

## Running the experiments
Examples on how to run the training and inference can be found in `exp` folder. Before running the commands as specified below, add the path to the scripts to your `PYTHONPATH`
```
export PYTHONPATH=<path-to-the-repository>/src:$PYTHONPATH
```

### Training VAE and noise models
This section can be skipped if you are using the pre-trained models. Otherwise, to train the VAE, you can run
```
cd exp/vae
python -m run
```
The script `run.py` serves both as the configuration file for the experiment and the launch of the training. The script will results in several created directories in `exp/vae`. Directory `models` will contain the trained model. If you want to change the paths to the output directories or the input data, please check the configuration in `run.py`.

Similarly, to train the noise models
```
cd exp/noise_model
python -m run
```
The directory `exp/noise_model/models` will contain the trained models.

### VAE-DOLPHIN inference
```
cd exp/infer_vae_noise15-20
python -m run with i_split=0 n_split=600
python compute_sdr.py
```
- The call above runs the inference on 1/600 partition of the data. By removing `i_split` and `n_split` arguments, you can run it on the entire dataset. This will however take a long time, we thus recommend parallelization, for which you can make use of `i_split` (index of the partition) and `n_split` (number of partitions). The example of how to use these with SGE cluster can be found in `exp/infer_vae_noise15-20/run.sh`.
- The call above corresponds to experiment with VAE-DOLPHIN with noise in SNR range 15-20 dB. For other experiments, see `exp/infer_*`, e.g. `exp/infer_vae_noise10-15` for different level of noise, or `exp/infer_gmm_noise15-20` for GMM-DOLPHIN experiment.
- The scripts creates several directories in `exp/infer_vae_noise15-20`. If you want to change the paths to the output directories or the input data, please check the configuration in `run.py`.
- The `compute_sdr.py` script will compute SDR, and write into `Out/sdr/tt/per_utt` and `Out/sdr/tt/avg`. Note that the computed number is absolute SDR, to obtain SDR improvements, subtract the mixture SDR value (available in the paper or in [RESULTS_VAE.md](RESULTS_VAE.md)).


## Results
Folowing table shows the schieved performance as presented in the paper. The metric is SDR improvement compared to SDR of the original mixtures.

| noise SNR [dB] | 15-20 | 10-15 | 5-10 | 0-5  | -5-0 |                                             |
|----------------|-------|-------|------|------|------|---------------------------------------------|
| GMM-DOLPHIN    | 12.1  | 10.6  | 8.8  | 7.3  | 6.8  | `exp/infer_gmm_noise<lowSNR>-<highSNR>`     |
| VAE-DOLPHIN    | 12.7  | 11.4  | 10.0 | 8.9  | 8.1  | `exp/infer_vae_noise<lowSNR>-<highSNR>`     |

The paper contains also results with PIT, for which we used the code in [uPIT-for-speech-separation](https://github.com/funcwj/uPIT-for-speech-separation) repository. The scripts in `exp/infer_pit_*` show how to get the results using the masks obtained by PIT.

The VAE results are averaged over runs of 5 different models. The standard deviation is around 0.2 dB. We suspect that with longer training, the different models would converge to results with smaller deviation. For the full results of all runs, see [RESULTS_VAE.md](RESULTS_VAE.md).

## License
See [LICENSE.txt](LICENSE.txt) for conditions under which the code can be used. If you use the code, please consider citing the paper
```
@INPROCEEDINGS{Zmolikova21VAEDolphin,
  author={K. {Zmolikova} and M. {Delcroix} and L. {Burget} and T. {Nakatani} and J. {\v{C}ernock\'{y}}},
  booktitle={2021 IEEE Spoken Language Technology Workshop (SLT)}, 
  title={Integration of variational autoencoder and spatial clustering for adaptive multi-channel neural speech separation}, 
  year={2021}}
```
