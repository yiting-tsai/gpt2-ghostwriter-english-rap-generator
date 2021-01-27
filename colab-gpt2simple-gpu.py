# Reference
# https://colab.research.google.com/drive/1VLG8e7YSEwypxU-noRNhsv5dW4NfTGce#scrollTo=P8wSlgXoDPCR

# SET UP
## Tensorflow 1.14 or 1.15
%tensorflow_version 1.x
!pip install -q gpt-2-simple
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import gpt_2_simple as gpt2
from google.colab import files
#uploaded = files.upload()


# CHECK GPU
!nvidia-smi

# MODEL 
gpt2.download_gpt2(model_name="355M") #124M

sess = gpt2.start_tf_sess()

gpt2.finetune(sess,
              dataset='lyrics.txt',
              model_name='355M',
              steps=1500,
              restore_from='fresh',
              run_name='run1',
              print_every=100,
              sample_every=500,
              save_every=500
              )


# SAVE MODEL
## to GoogleDrive
gpt2.mount_gdrive()
gpt2.copy_checkpoint_to_gdrive(run_name='run1')

## locally
!zip -r models.zip ./models
files.download("models.zip")

!zip -r checkpoint.zip ./checkpoint
files.download("checkpoint.zip")

!zip -r samples.zip ./samples
files.download("samples.zip")


# INFERENCE
gpt2.generate(sess,
              run_name='run1',
              length=250,
              temperature=0.7,
              prefix="I came I saw I praise the Lord",
              top_p=0.9,
              top_k=50
              #nsamples=5,
              #batch_size=5
              )