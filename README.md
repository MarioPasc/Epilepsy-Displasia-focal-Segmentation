# Epilepsy Lesion Segmentation using YOLOv8

This project focuses on employing the YOLOv8 neural network architecture for segmenting epilepsy lesions, specifically focal cortical dysplasia type II, in MRI images. The dataset used for training and evaluation is the "An open presurgery MRI dataset of people with epilepsy and focal cortical dysplasia type II" available on OpenNeuro.

## Dataset Information

The dataset citation is as follows:

```
@dataset{ds004199:1.0.5,
  author = {Fabiane Schuch and Lennart Walger and Matthias Schmitz and Bastian David and Tobias Bauer and Antonia Harms and Laura Fischbach and Freya Schulte and Martin Schidlowski and Johannes Reiter and Felix Bitzer and Randi von Wrede and Attila Rácz and Tobias Baumgartner and Valeri Borger and Matthias Schneider and Achim Flender and Albert Becker and Hartmut Vatter and Bernd Weber and Louisa Specht-Riemenschneider and Alexander Radbruch and Rainer Surges and Theodor Rüber},
  title = {"An open presurgery MRI dataset of people with epilepsy and focal cortical dysplasia type II"},
  year = {2023},
  doi = {doi:10.18112/openneuro.ds004199.v1.0.5},
  publisher = {OpenNeuro}
}
```

## Project Structure

- `src/`: Source code directory.
  - `data_processing.py`: Python script for converting and preprocessing images.
  - `model_training.py`: Python script for training the YOLOv8 model.
  - `model_evaluation.py`: Python script for evaluating the trained model on test data.
- `README.md`: This file, providing an overview of the project.

## Dependencies

- Python 3.8.18
- The following packages:

```bash
_libgcc_mutex             0.1                        main  
_openmp_mutex             5.1                       1_gnu  
anyio                     4.2.0            py38h06a4308_0  
argon2-cffi               21.3.0             pyhd3eb1b0_0  
argon2-cffi-bindings      21.2.0           py38h7f8727e_0  
arrow                     1.3.0                    pypi_0    pypi
asttokens                 2.0.5              pyhd3eb1b0_0  
async-lru                 2.0.4            py38h06a4308_0  
attrs                     23.1.0           py38h06a4308_0  
babel                     2.11.0           py38h06a4308_0  
backcall                  0.2.0              pyhd3eb1b0_0  
beautifulsoup4            4.12.2           py38h06a4308_0  
blas                      1.0                         mkl  
bleach                    4.1.0              pyhd3eb1b0_0  
brotli                    1.0.9                h5eee18b_7  
brotli-bin                1.0.9                h5eee18b_7  
brotli-python             1.0.9            py38h6a678d5_7  
ca-certificates           2024.2.2             hbcca054_0    conda-forge
certifi                   2023.7.22                pypi_0    pypi
cffi                      1.16.0           py38h5eee18b_0  
chardet                   5.2.0                    pypi_0    pypi
charset-normalizer        3.3.2                    pypi_0    pypi
cmake                     3.27.7                   pypi_0    pypi
comm                      0.2.1                    pypi_0    pypi
contourpy                 1.0.5            py38hdb19cb5_0  
cryptography              41.0.7           py38hdda0065_0  
cuda-cccl                 11.8.89                       0    nvidia/label/cuda-11.8.0
cuda-command-line-tools   11.8.0                        0    nvidia/label/cuda-11.8.0
cuda-compiler             11.8.0                        0    nvidia/label/cuda-11.8.0
cuda-cudart               11.8.89                       0    nvidia/label/cuda-11.8.0
cuda-cudart-dev           11.8.89                       0    nvidia/label/cuda-11.8.0
cuda-cuobjdump            11.8.86                       0    nvidia/label/cuda-11.8.0
cuda-cupti                11.8.87                       0    nvidia/label/cuda-11.8.0
cuda-cuxxfilt             11.8.86                       0    nvidia/label/cuda-11.8.0
cuda-documentation        11.8.86                       0    nvidia/label/cuda-11.8.0
cuda-driver-dev           11.8.89                       0    nvidia/label/cuda-11.8.0
cuda-gdb                  11.8.86                       0    nvidia/label/cuda-11.8.0
cuda-libraries            11.8.0                        0    nvidia/label/cuda-11.8.0
cuda-libraries-dev        11.8.0                        0    nvidia/label/cuda-11.8.0
cuda-memcheck             11.8.86                       0    nvidia/label/cuda-11.8.0
cuda-nsight               11.8.86                       0    nvidia/label/cuda-11.8.0
cuda-nsight-compute       11.8.0                        0    nvidia/label/cuda-11.8.0
cuda-nvcc                 11.8.89                       0    nvidia/label/cuda-11.8.0
cuda-nvdisasm             11.8.86                       0    nvidia/label/cuda-11.8.0
cuda-nvml-dev             11.8.86                       0    nvidia/label/cuda-11.8.0
cuda-nvprof               11.8.87                       0    nvidia/label/cuda-11.8.0
cuda-nvprune              11.8.86                       0    nvidia/label/cuda-11.8.0
cuda-nvrtc                11.8.89                       0    nvidia/label/cuda-11.8.0
cuda-nvrtc-dev            11.8.89                       0    nvidia/label/cuda-11.8.0
cuda-nvtx                 11.8.86                       0    nvidia/label/cuda-11.8.0
cuda-nvvp                 11.8.87                       0    nvidia/label/cuda-11.8.0
cuda-profiler-api         11.8.86                       0    nvidia/label/cuda-11.8.0
cuda-sanitizer-api        11.8.86                       0    nvidia/label/cuda-11.8.0
cuda-toolkit              11.8.0                        0    nvidia/label/cuda-11.8.0
cuda-tools                11.8.0                        0    nvidia/label/cuda-11.8.0
cuda-visual-tools         11.8.0                        0    nvidia/label/cuda-11.8.0
cycler                    0.11.0             pyhd3eb1b0_0  
cyrus-sasl                2.1.28               h52b45da_1  
dbus                      1.13.18              hb2f20db_0  
debugpy                   1.6.7            py38h6a678d5_0  
decorator                 5.1.1              pyhd3eb1b0_0  
defusedxml                0.7.1              pyhd3eb1b0_0  
exceptiongroup            1.2.0            py38h06a4308_0  
executing                 0.8.3              pyhd3eb1b0_0  
expat                     2.5.0                h6a678d5_0  
filelock                  3.13.1                   pypi_0    pypi
fontconfig                2.14.1               h4c34cd2_2  
fonttools                 4.25.0             pyhd3eb1b0_0  
fqdn                      1.5.1                    pypi_0    pypi
freetype                  2.12.1               h4a9f257_0  
gds-tools                 1.4.0.31                      0    nvidia/label/cuda-11.8.0
giflib                    5.2.1                h5eee18b_3  
glib                      2.69.1               he621ea3_2  
gst-plugins-base          1.14.1               h6a678d5_1  
gstreamer                 1.14.1               h5eee18b_1  
h11                       0.14.0                   pypi_0    pypi
httpcore                  1.0.4                    pypi_0    pypi
httpx                     0.27.0                   pypi_0    pypi
icu                       73.1                 h6a678d5_0  
idna                      3.4              py38h06a4308_0  
importlib-metadata        7.0.1            py38h06a4308_0  
importlib_metadata        7.0.1                hd3eb1b0_0  
importlib_resources       6.1.1            py38h06a4308_1  
intel-openmp              2023.1.0         hdb19cb5_46306  
ipykernel                 6.28.0           py38h06a4308_0  
ipython                   8.12.2           py38h06a4308_0  
ipywidgets                8.1.2                    pypi_0    pypi
isoduration               20.11.0                  pypi_0    pypi
jedi                      0.18.1           py38h06a4308_1  
jinja2                    3.1.2                    pypi_0    pypi
jpeg                      9e                   h5eee18b_1  
json5                     0.9.6              pyhd3eb1b0_0  
jsonpointer               2.4                      pypi_0    pypi
jsonschema                4.19.2           py38h06a4308_0  
jsonschema-specifications 2023.7.1         py38h06a4308_0  
jupyter                   1.0.0            py38h06a4308_8  
jupyter-lsp               2.2.0            py38h06a4308_0  
jupyter_client            8.6.0            py38h06a4308_0  
jupyter_console           6.6.3            py38h06a4308_0  
jupyter_core              5.5.0            py38h06a4308_0  
jupyter_events            0.8.0            py38h06a4308_0  
jupyter_server            2.10.0           py38h06a4308_0  
jupyter_server_terminals  0.4.4            py38h06a4308_1  
jupyterlab                4.1.3                    pypi_0    pypi
jupyterlab-widgets        3.0.10                   pypi_0    pypi
jupyterlab_pygments       0.1.2                      py_0  
jupyterlab_server         2.25.1           py38h06a4308_0  
kiwisolver                1.4.4            py38h6a678d5_0  
krb5                      1.20.1               h143b758_1  
lcms2                     2.12                 h3be6417_0  
ld_impl_linux-64          2.38                 h1181459_1  
lerc                      3.0                  h295c915_0  
libbrotlicommon           1.0.9                h5eee18b_7  
libbrotlidec              1.0.9                h5eee18b_7  
libbrotlienc              1.0.9                h5eee18b_7  
libclang                  14.0.6          default_hc6dbbc7_1  
libclang13                14.0.6          default_he11475f_1  
libcublas                 11.11.3.6                     0    nvidia/label/cuda-11.8.0
libcublas-dev             11.11.3.6                     0    nvidia/label/cuda-11.8.0
libcufft                  10.9.0.58                     0    nvidia/label/cuda-11.8.0
libcufft-dev              10.9.0.58                     0    nvidia/label/cuda-11.8.0
libcufile                 1.4.0.31                      0    nvidia/label/cuda-11.8.0
libcufile-dev             1.4.0.31                      0    nvidia/label/cuda-11.8.0
libcups                   2.4.2                h2d74bed_1  
libcurand                 10.3.0.86                     0    nvidia/label/cuda-11.8.0
libcurand-dev             10.3.0.86                     0    nvidia/label/cuda-11.8.0
libcusolver               11.4.1.48                     0    nvidia/label/cuda-11.8.0
libcusolver-dev           11.4.1.48                     0    nvidia/label/cuda-11.8.0
libcusparse               11.7.5.86                     0    nvidia/label/cuda-11.8.0
libcusparse-dev           11.7.5.86                     0    nvidia/label/cuda-11.8.0
libdeflate                1.17                 h5eee18b_1  
libedit                   3.1.20230828         h5eee18b_0  
libffi                    3.4.4                h6a678d5_0  
libgcc-ng                 11.2.0               h1234567_1  
libgomp                   11.2.0               h1234567_1  
libllvm14                 14.0.6               hdb19cb5_3  
libnpp                    11.8.0.86                     0    nvidia/label/cuda-11.8.0
libnpp-dev                11.8.0.86                     0    nvidia/label/cuda-11.8.0
libnvjpeg                 11.9.0.86                     0    nvidia/label/cuda-11.8.0
libnvjpeg-dev             11.9.0.86                     0    nvidia/label/cuda-11.8.0
libpng                    1.6.39               h5eee18b_0  
libpq                     12.17                hdbd6064_0  
libsodium                 1.0.18               h7b6447c_0  
libstdcxx-ng              11.2.0               h1234567_1  
libtiff                   4.5.1                h6a678d5_0  
libuuid                   1.41.5               h5eee18b_0  
libwebp                   1.3.2                h11a3e52_0  
libwebp-base              1.3.2                h5eee18b_0  
libxcb                    1.15                 h7f8727e_0  
libxkbcommon              1.0.1                h5eee18b_1  
libxml2                   2.10.4               hf1b16e4_1  
lit                       17.0.4                   pypi_0    pypi
lz4-c                     1.9.4                h6a678d5_0  
markupsafe                2.1.3            py38h5eee18b_0  
matplotlib                3.7.2            py38h06a4308_0  
matplotlib-base           3.7.2            py38h1128e8f_0  
matplotlib-inline         0.1.6            py38h06a4308_0  
mistune                   2.0.4            py38h06a4308_0  
mkl                       2023.1.0         h213fc3f_46344  
mkl-service               2.4.0            py38h5eee18b_1  
mkl_fft                   1.3.8            py38h5eee18b_0  
mkl_random                1.2.4            py38hdb19cb5_0  
mpmath                    1.3.0                    pypi_0    pypi
munkres                   1.1.4                      py_0  
mysql                     5.7.24               h721c034_2  
nbclient                  0.8.0            py38h06a4308_0  
nbconvert                 7.10.0           py38h06a4308_0  
nbformat                  5.9.2            py38h06a4308_0  
ncurses                   6.4                  h6a678d5_0  
nest-asyncio              1.5.6            py38h06a4308_0  
networkx                  3.1                      pypi_0    pypi
nibabel                   5.2.1                    pypi_0    pypi
ninja                     1.11.1.1                 pypi_0    pypi
notebook                  7.1.1                    pypi_0    pypi
notebook-shim             0.2.3            py38h06a4308_0  
nsight-compute            2022.3.0.22                   0    nvidia/label/cuda-11.8.0
numpy                     1.24.4                   pypi_0    pypi
numpy-base                1.24.3           py38h060ed82_1  
opencv-python             4.2.0.32                 pypi_0    pypi
openjpeg                  2.4.0                h3ad879b_0  
openssl                   3.0.13               h7f8727e_0  
overrides                 7.4.0            py38h06a4308_0  
packaging                 23.1             py38h06a4308_0  
pandocfilters             1.5.0              pyhd3eb1b0_0  
parso                     0.8.3              pyhd3eb1b0_0  
pcre                      8.45                 h295c915_0  
pexpect                   4.8.0              pyhd3eb1b0_3  
pickleshare               0.7.5           pyhd3eb1b0_1003  
pillow                    10.1.0                   pypi_0    pypi
pip                       24.0                     pypi_0    pypi
pkgutil-resolve-name      1.3.10           py38h06a4308_1  
platformdirs              3.10.0           py38h06a4308_0  
ply                       3.11                     py38_0  
prometheus_client         0.14.1           py38h06a4308_0  
prompt-toolkit            3.0.43           py38h06a4308_0  
prompt_toolkit            3.0.43               hd3eb1b0_0  
psutil                    5.9.0            py38h5eee18b_0  
ptyprocess                0.7.0              pyhd3eb1b0_2  
pure_eval                 0.2.2              pyhd3eb1b0_0  
pycparser                 2.21               pyhd3eb1b0_0  
pygments                  2.15.1           py38h06a4308_1  
pyopenssl                 23.2.0           py38h06a4308_0  
pyparsing                 3.0.9            py38h06a4308_0  
pyqt                      5.15.10          py38h6a678d5_0  
pyqt5-sip                 12.13.0          py38h5eee18b_0  
pysocks                   1.7.1            py38h06a4308_0  
python                    3.8.18               h955ad1f_0  
python-dateutil           2.8.2              pyhd3eb1b0_0  
python-fastjsonschema     2.16.2           py38h06a4308_0  
python-json-logger        2.0.7            py38h06a4308_0  
pytz                      2023.3.post1     py38h06a4308_0  
pyyaml                    6.0.1            py38h5eee18b_0  
pyzmq                     25.1.2           py38h6a678d5_0  
qt-main                   5.15.2              h53bd1ea_10  
qtconsole                 5.5.0            py38h06a4308_0  
qtpy                      2.4.1            py38h06a4308_0  
readline                  8.2                  h5eee18b_0  
referencing               0.30.2           py38h06a4308_0  
requests                  2.31.0           py38h06a4308_0  
rfc3339-validator         0.1.4            py38h06a4308_0  
rfc3986-validator         0.1.1            py38h06a4308_0  
rpds-py                   0.10.6           py38hb02cf49_0  
send2trash                1.8.2            py38h06a4308_0  
setuptools                68.2.2                   pypi_0    pypi
sip                       6.7.12           py38h6a678d5_0  
six                       1.16.0             pyhd3eb1b0_1  
sniffio                   1.3.0            py38h06a4308_0  
soupsieve                 2.5              py38h06a4308_0  
sqlite                    3.41.2               h5eee18b_0  
stack_data                0.2.0              pyhd3eb1b0_0  
sympy                     1.12                     pypi_0    pypi
tbb                       2021.8.0             hdb19cb5_0  
terminado                 0.17.1           py38h06a4308_0  
tinycss2                  1.2.1            py38h06a4308_0  
tinycudann                1.7                      pypi_0    pypi
tk                        8.6.12               h1ccaba5_0  
tomli                     2.0.1            py38h06a4308_0  
torch                     2.0.1+cu118              pypi_0    pypi
torchvision               0.15.2+cu118             pypi_0    pypi
tornado                   6.3.3            py38h5eee18b_0  
traitlets                 5.7.1            py38h06a4308_0  
triton                    2.0.0                    pypi_0    pypi
types-python-dateutil     2.8.19.20240106          pypi_0    pypi
typing-extensions         4.8.0                    pypi_0    pypi
typing_extensions         4.9.0            py38h06a4308_1  
uri-template              1.3.0                    pypi_0    pypi
urllib3                   2.0.7                    pypi_0    pypi
wcwidth                   0.2.5              pyhd3eb1b0_0  
webcolors                 1.13                     pypi_0    pypi
webencodings              0.5.1                    py38_1  
websocket-client          0.58.0           py38h06a4308_4  
wheel                     0.41.2           py38h06a4308_0  
widgetsnbextension        4.0.10                   pypi_0    pypi
xz                        5.4.2                h5eee18b_0  
yaml                      0.2.5                h7b6447c_0  
zeromq                    4.3.5                h6a678d5_0  
zipp                      3.17.0           py38h06a4308_0  
zlib                      1.2.13               h5eee18b_0  
zstd                      1.5.5                hc292b87_0  
``
