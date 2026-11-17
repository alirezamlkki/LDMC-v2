# Quantized Sampling Distillation for Efficient Uncertainty Quantification

Code for the paper "Quantized Sampling Distillation for Efficient Uncertainty Quantification" (under review).

LDMC is a teacher–student framework for uncertainty estimation. An MC-based teacher
(MC Dropout or MC Laplace approximation) is quantized into a small set of control points,
and the student receives these control points through soft dropout layers.
This repository contains the two instantiations, QMCD (quantized MC Dropout) and
QMCLA (quantized MC Laplace approximation), and the DDU baseline.

## Usage

Run the scripts from inside each experiment folder, since the modules are imported by name.

SVHN uses separate scripts for the teacher-side evaluation and the student:

```
cd svhn/qmcd
python teacher_chart_qmcd.py <p_drop> <p_rw>          # train and evaluate the QMCD teacher
python main_student.py <p_drop> <p_rw> <True|False|rand>

cd svhn/qmcla
python teacher_chart.py                                # train and evaluate the QMCLA teacher
python main_student.py <p_m> <True|False|rand>
```

The last argument selects the student: `True` is LDMC, `False` is standard single-pass
distillation, and `rand` is the ablation with randomly selected control points.

## Setup

```
pip install -r requirements.txt
```

MNIST, CIFAR-10 and SVHN are downloaded by torchvision into `./data`,
DBpedia-14 is downloaded with Hugging Face `datasets`, and Ambiguous-MNIST with `ddu_dirty_mnist`.

Before the first run, create the checkpoint folder inside the experiment folder:

```
mkdir -p model
python main.py
```