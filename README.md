## Rethinking Node-wise Propagation for Large-scale Graph Learning

### **Requirements**

Hardware environment: Intel(R) Xeon(R) Gold 6230R CPU @ 2.10GHz, NVIDIA GeForce RTX 3090 with 24GB memory.

Software environment: Ubuntu 18.04.6, Python 3.9, PyTorch 1.11.0 and CUDA 11.8.

1. Please refer to [PyTorch](https://pytorch.org/get-started/locally/) and [PyG](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) to install the environments;
2. Run 'pip install -r requirements.txt' to download required packages;

### **Training**

To train the model(s) in the paper

1. Please put the data under the dataset/homo_data, we have already downloaded cora as examples.
2. Open main.py to run our program with our ATP plug-in. We need to generate `degree_centrality`, `clustering_coefficients`, `Engienvector_centrality`.

for example, if you want to run cora under sgc model, just try the command below

```bash
python main.py --prop_steps 10 --model_name sgc --data_name cora --r_way together --lr 0.2 --dropout 0.0 --weight_decay 1e-5 --num_epochs 150 --a 0.3 --b 0.7 --c 0.0 --normalize_times 10 --gpu_id 1
```

