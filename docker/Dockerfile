# docker build --tag=torch.py .
FROM pytorch/pytorch:latest

RUN apt update \
&& apt install -y \
build-essential \
git \
&& rm -rf /var/lib/apt/lists/*

# Install kornia first, otherwise it uninstalls the pytorch-nightly again (?)
# RUN pip install kornia
RUN pip install git+https://github.com/adriankoering/kornia.git@resample_border
# RUN pip install git+https://github.com/kornia/kornia

# RUN conda install pytorch torchvision cudatoolkit=10.2 -c pytorch-nightly
RUN conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
RUN conda install -c conda-forge pytorch-lightning
RUN pip install 'ray[tune]' moviepy av wandb hyperopt nonechucks

RUN pip uninstall -y pillow
RUN CC="cc -mavx2" pip install -U --force-reinstall pillow-simd

# Install Jupyer Notebooks
RUN pip install matplotlib jupyter
RUN mkdir /.local && chmod a+rwx /.local && mkdir /.jupyter && chmod a+rwx /.jupyter

# Install Nvidia Dali
RUN pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda100
ENV NVIDIA_DRIVER_CAPABILITIES compute,video,utility

RUN pip install tensorflow==2.2.0 tensorflow-datasets
RUN pip install scikit-image opencv-python-headless

EXPOSE 6006
EXPOSE 8265
EXPOSE 8888

# CMD ["bash" , "-c", "jupyter notebook --ip 0.0.0.0 --no-browser --allow-root"]
CMD ["bash"]
