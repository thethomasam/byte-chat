from datasets import load_dataset

dataset = load_dataset("AnishJoshi/nl2bash-custom")
print(dataset['train']['bash_code'][0])