Tested on Ubuntu. Version 1.00 with bugs 

# Workflow

1. Clone the Github repo
```
git clone https://github.com/nazarb/archeo_vision.git
```
2. Change directory to archeo_vision

3. Change director to vision_pipeline

4.  
## First, pull the smaller model
```
sudo docker exec vision-ollama ollama pull qwen2.5-vl:7b

```

## Or if you want qwen3-vl:8b (check if it exists)

```
sudo docker exec vision-ollama ollama pull qwen3-vl:8b
```
