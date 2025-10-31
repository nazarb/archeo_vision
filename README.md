Tested on Ubuntu. Version 1.00 with bugs 

# Workflow

1. Clone the Github repo
```
git clone https://github.com/nazarb/archeo_vision.git
```
2. Change directory to archeo_vision

```
cd  archeo_vision
```

4. Change director to vision_pipeline
```
cd vision_pipeline
```
5. Create a virtual enviroment on Ubuntu

First, find Python 3.11:
```
which python3.11
```
If it exists, create a proper venv:
bash# Remove the wrong venv
```
rm -rf venv311
```

# Create with Python 3.11 explicitly
```
/usr/bin/python3.11 -m venv venv311
source venv311/bin/activate
```
# Verify you're using 3.11
```
python --version
```

6. Install the requirements in the venv311
```
pip install -r requirements.txt
```
7. Install the Docker compose
```
sudo apt-get update
# sudo apt-get install docker-compose-plugin # 
sudo apt-get install docker compose # I was using this
```
8.  Compose Docker

```
# Stop all services to clear GPU memory
sudo docker compose down

# Start them again
sudo docker compose up -d

# Wait for them to be ready
sleep 30

# Check they're healthy
curl http://localhost:8080/health
```

6.  Install selected Ollama model

```
sudo docker exec vision-ollama ollama pull qwen2.5-vl:7b
```

```
sudo docker exec vision-ollama ollama pull qwen3-vl:8b
```
7. Try the pipeline. Change the directory to the archaeo_vision
```
cd -
cd archaeo-vision
```
8. Run first the detection of the labels. You have to put the images into archaeo_shared/images folder
```
python archeo_vision_client.py --model qwen2.5-vl:7b

```
9. Now run, the code that going to change the names according to results of previous phase
```
python archeo_file_organizer.py --create-index

```

