# Reference:
# https://towardsdatascience.com/text-generation-gpt-2-lstm-markov-chain-9ea371820e1e#3c34
# https://github.com/klaudia-nazarko/nlg-text-generation/blob/main/gpt_2.ipynb
# https://huggingface.co/transformers/model_doc/gpt2.html   
# https://colab.research.google.com/github/philschmid/fine-tune-GPT-2/blob/master/Fine_tune_a_non_English_GPT_2_Model_with_Huggingface.ipynb#scrollTo=VCaunLMtlPfw
# https://github.com/huggingface/blog/blob/master/notebooks/02_how_to_generate.ipynb

# IMPORT DATASET
## choose train.txt and eval.txt two files
from google.colab import files

uploaded = files.upload()

for fn in uploaded.keys():
  print('User uploaded file "{name}" with length {length} bytes'.format(
      name=fn, length=len(uploaded[fn])))


# SET UP TPU
!pip install cloud-tpu-client==0.10 https://storage.googleapis.com/tpu-pytorch/wheels/torch_xla-1.7-cp36-cp36m-linux_x86_64.whl

%tensorflow_version 2.x
import tensorflow as tf
print("Tensorflow version " + tf.__version__)

try:
  tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
  print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
except ValueError:
  raise BaseException('ERROR: Not connected to a TPU runtime; please see the previous cell in this notebook for instructions!')

tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)
tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)


# PIP INSTALL PACKAGE & IMPORT LIBRARIES
!pip install transformers #!pip install git+https://github.com/huggingface/transformers.git
!pip install datasets

VERSION = "20200325"  #@param ["1.5" , "20200325", "nightly"]
!curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py
!python pytorch-xla-env-setup.py --version $VERSION

# import torch_xla.distributed.xla_multiprocessing as xmp
# import json

#args_dict = {
#  "num_cores": 8,
#  'training_script': 'train_gpt2_rap.py',
#  "model_name_or_path": 'gpt2-medium',
#  "max_len": 512 ,
#  "target_max_len": 16,
#  "OUTPUT_DIR": '/models/tpu/',
#  "output_dir": '/models/tpu/',
#  "overwrite_output_dir": True,
#  "per_gpu_train_batch_size": 8,
#  "per_gpu_eval_batch_size": 8,
#  "gradient_accumulation_steps": 4,
#  "learning_rate": 1e-4,
#  "tpu_num_cores": 8,
#  "num_train_epochs": 4,
#  "do_train": True
#}

#with open('args.json', 'w') as f:
#  json.dump(args_dict, f)


# MODEL
from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

train_path = 'train.txt'
test_path = 'eval.txt'

from transformers import TextDataset,DataCollatorForLanguageModeling

def load_dataset(train_path,test_path,tokenizer):
    train_dataset = TextDataset(
          tokenizer=tokenizer,
          file_path=train_path,
          block_size=128)
     
    test_dataset = TextDataset(
          tokenizer=tokenizer,
          file_path=test_path,
          block_size=128)   
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )
    return train_dataset,test_dataset,data_collator

train_dataset,test_dataset,data_collator = load_dataset(train_path,test_path,tokenizer)
#print(tokenizer.decode(train_dataset[5]))

## Initialize Trainer with TrainingArguments and GPT-2 model
from transformers import GPT2LMHeadModel
model = GPT2LMHeadModel.from_pretrained('gpt2')

from transformers import Trainer, TrainingArguments,AutoModelWithLMHead
training_args = TrainingArguments(
    output_dir="model/out", #The output directory
    overwrite_output_dir=True, #overwrite the content of the output directory
    num_train_epochs=200, # number of training epochs
    per_device_train_batch_size=32, # batch size for training
    per_device_eval_batch_size=32,  # batch size for evaluation
    learning_rate = 5e-5, # defaults to 5e-5
    eval_steps = 400, # Number of update steps between two evaluations.
    save_steps=800, # after # steps model is saved 
    warmup_steps=500,# number of warmup steps for learning rate scheduler
    )


trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    #prediction_loss_only=True,
)

## FINE-TUNE
trainer.train()
trainer.save_model()


# DOWNLOAD MODEL WEIGHTS
!zip -r model.zip ./model

from google.colab import files
files.download("model.zip")

# !python transformers/examples/xla_spawn.py --num_cores 8 \
#	run_clm.py \
#    --output_dir=/models/tpu \
#    --model_type=gpt2 \
#    --model_name_or_path=gpt2-medium \
#    --train_file train.txt\
#    --validation_file eval.txt\
#    --do_train \
#    --do_eval \
#    --per_device_train_batch_size 2\
#    --overwrite_output_dir

#xmp.spawn(_mp_fn, args=(), nprocs=8, start_method='fork')


# INFERENCE
# print(trainer.args.device)
model = GPT2LMHeadModel.from_pretrained('model/out').to('cpu') # because its loaded on xla by default
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

start_prompt="Percocets, molly, Percocets"
inputs=tokenizer.encode(start_prompt, add_special_tokens=False, return_tensors="pt")

#prompt_length = len(tokenizer.decode(inputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True))
outputs = model.generate(inputs, max_length=250, do_sample=True, top_p=0.9, top_k=60, temperature=0.85)
generated = tokenizer.decode(outputs[0])#[prompt_length:]
print(generated)