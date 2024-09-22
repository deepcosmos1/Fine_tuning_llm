# Use NVIDIA PyTorch image as the base image
FROM nvcr.io/nvidia/pytorch:24.02-py3

# Install SSH server
RUN apt-get update && apt-get install -y openssh-server
RUN mkdir /var/run/sshd

# Configure SSH for passwordless authentication
RUN sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config
RUN sed -i 's/#PubkeyAuthentication yes/PubkeyAuthentication yes/' /etc/ssh/sshd_config
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# Add your public SSH key for passwordless SSH login
RUN mkdir /root/.ssh
RUN echo "" > /root/.ssh/authorized_keys

ENV SSH_PORT=22
RUN cat /etc/ssh/sshd_config > ${STAGE_DIR}/sshd_config && \
        sed "0,/^Port 22/s//Port ${SSH_PORT}/" ${STAGE_DIR}/sshd_config > /etc/ssh/sshd_config

# Ensure correct permissions
RUN chmod 700 /root/.ssh && chmod 600 /root/.ssh/authorized_keys

#pip install requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

# Copy your Conda environment files
COPY environment.yml .
COPY environmentInference.yml .

# Create Conda environments
RUN conda env create -f environment.yml
RUN conda env create -f environmentInference.yml
RUN echo "source activate myvenv" >> ~/.bashrc

# Set the working directory
WORKDIR /src

# Copy the application files
COPY . .

# Expose the ports for Gradio and SSH
EXPOSE 7860
EXPOSE 2222
EXPOSE 5000

# Start SSH server and a shell session with the activated conda environment
CMD service ssh start && /bin/bash -i
