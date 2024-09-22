# DMI Generative AI Platform

### Structured Synthetic Data generation using Large Language Models

### Run Locally Using Docker

To run the DMI Generative AI Platform locally using Docker, follow these steps:

1. Clone the repository:

```
git clone https://github.com/DMI-Finance/finetuning-experiments.git
```

2. Change to the `finetuning-experiments` directory:

```
cd finetuning-experiments
```

3. Build the Docker image:

```
docker build -t app .
```

4. Run the Docker container:

```
docker run -d -p 7860:7860 --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it app
```

This will start the container in detached mode and make the Gradio app available on port 7860.

5. Connect to the running container:

```
docker attach <container_id>
```

Replace `<container_id>` with the ID of the running container.

6. Set up the environment:

```
python setup.py
```

This will download the necessary datasets, tools, and libraries.

7. Run the test script to start the Gradio app:

```
python test.py
```

The Gradio app will be available at the following URL:

```
http://<Public IPv4 DNS>:7860
```

Replace `<Public IPv4 DNS>` with the public DNS of your EC2 instance.
